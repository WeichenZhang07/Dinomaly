import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Dict, Callable

__all__ = ["de_convnext_dinov3_base"]


# ================= ConvNeXt Block =================
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return x + shortcut


class TransformerBlock(nn.Module):
    """
    Hybrid Transformer block with Linear Attention (no softmax).
    - 输入 / 输出: (B, C, H, W)
    - 支持可选相对位置编码 RPE
    - 支持 multi-head，注意力为 φ(Q)(φ(K)^T V) 形式
    """

    class RelPosEnc(nn.Module):
        """Learnable relative positional encoding with adaptive interpolation."""

        def __init__(self, dim, max_hw=32, scale_init=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1) * scale_init)
            self.rel_bias = nn.Parameter(torch.randn(dim, max_hw, max_hw))

        def forward(self, x, H, W):
            # interpolate bias to match feature resolution
            bias = F.interpolate(
                self.rel_bias.unsqueeze(0),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)  # (C,H,W)
            return x + self.scale * bias.unsqueeze(0)  # (B,C,H,W)

    def __init__(
        self,
        dim,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        layer_scale_init_value: float = 1e-6,
        dropout: float = 0.0,
        use_cross_attention: bool = False,  # 先只实现 self-LA，cross 可后续加
        use_rpe: bool = True,
    ):
        super().__init__()

        assert (
            dim % num_heads == 0
        ), f"dim {dim} must be divisible by num_heads {num_heads}"

        self.use_rpe = use_rpe
        self.use_cross_attention = use_cross_attention
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        # ---- RPE ----
        if use_rpe:
            self.rpe = TransformerBlock.RelPosEnc(dim)

        # ---- Linear Attention QKV 投影 ----
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        # ---- FeedForward ----
        hidden_dim = int(dim * mlp_ratio)
        self.norm_attn = nn.LayerNorm(dim, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # ---- LayerScale ----
        self.gamma_attn = nn.Parameter(
            torch.ones(dim) * layer_scale_init_value
        )
        self.gamma_ffn = nn.Parameter(
            torch.ones(dim) * layer_scale_init_value
        )

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        """Linear attention 激活 φ(x) = elu(x) + 1，确保正值。"""
        return F.elu(x) + 1.0

    def _linear_self_attn(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        return: (B, N, C)
        实现 LA(Q,K,V) = φ(Q) [ φ(K)^T V ]，带 multi-head。
        """
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B,N,3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # (3,B,h,N,d)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,h,N,d)

        # φ(Q), φ(K)
        q = self.phi(q)
        k = self.phi(k)

        # LA: φ(Q)(φ(K)^T V)
        # 先算 S = φ(K)^T V: (B,h,d,N) @ (B,h,N,d) -> (B,h,d,d)
        kv = torch.einsum("b h n d, b h n e -> b h d e", k, v)
        # 再 out = φ(Q) S: (B,h,N,d) @ (B,h,d,e) -> (B,h,N,e)
        out = torch.einsum("b h n d, b h d e -> b h n e", q, kv)

        out = self.attn_drop(out)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # (B,N,C)
        out = self.out_proj(out)  # 线性映射回 dim
        return out

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, C, H, W)
        当前只用 Linear Self-Attention，不使用 cross-attn。
        """
        B, C, H, W = x.shape

        # ---- 位置编码 ----
        if self.use_rpe:
            x = self.rpe(x, H, W)  # (B,C,H,W)

        # ---- 展平成 tokens ----
        x = x.flatten(2).transpose(1, 2)  # (B,N,C)

        # ---- Linear Self-Attention ----
        shortcut = x
        x_norm = self.norm_attn(x)
        attn_out = self._linear_self_attn(x_norm)  # (B,N,C)
        # LayerScale: gamma_attn: (C,) 广播到 (B,N,C)
        x = shortcut + self.gamma_attn * attn_out

        # ---- FFN ----
        shortcut2 = x
        x_norm2 = self.norm_ffn(x)
        ffn_out = self.mlp(x_norm2)
        x = shortcut2 + self.gamma_ffn * ffn_out

        # ---- 回到 (B,C,H,W) ----
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


# ================= Conv Block =================
class ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size: int = 3,
        layer_scale_init_value: float = 1e-6,
        use_norm: bool = True,
        use_activation: bool = True,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.use_norm = use_norm
        self.use_activation = use_activation

        # 用 GroupNorm(1, C) 来跟 LN/Transformer 风格对齐
        self.norm = nn.GroupNorm(1, dim) if use_norm else nn.Identity()

        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.act = nn.GELU() if use_activation else nn.Identity()

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv(x)
        x = self.act(x)
            # LayerScale 适配 BCHW
        gamma = self.gamma.view(1, -1, 1, 1)

        return shortcut + gamma * x


# ================= Block Registry =================
BLOCK_BUILDERS: Dict[str, Callable[[int, float], nn.Module]] = {
    "convnext": lambda dim, ls: ConvNeXtBlock(dim, layer_scale_init_value=ls),
    "conv": lambda dim, ls: ConvBlock(dim, layer_scale_init_value=ls),
    "transformer": lambda dim, ls: TransformerBlock(
        dim, layer_scale_init_value=ls, use_cross_attention=False
    ),
}


# ================= ReStage =================
class ReStage(nn.Module):
    """
    对称解码阶段：
      - 若干个 block（conv/convnext/transformer 按 pattern 排列）
      - 通道调整
      - 可选上采样（双线性插值 + Conv）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_types: List[str],  # e.g. ["convnext","transformer","conv","conv"]
        layer_scale_init_value: float = 1e-6,
        upsample: bool = True,
    ):
        super().__init__()

        # 动态构建 blocks：所有 block 输入输出通道均为 in_channels
        blocks = []
        for name in block_types:
            assert (
                name in BLOCK_BUILDERS
            ), f"Unknown block type: {name}. Supported: {list(BLOCK_BUILDERS.keys())}"
            blocks.append(BLOCK_BUILDERS[name](in_channels, layer_scale_init_value))
        self.blocks = nn.ModuleList(blocks)

        # 通道调整（例如 1024 -> 512 -> 128 -> 64）
        self.channel_reduce = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # 上采样：双线性插值 + Conv
        if upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            self.upsample = None

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        返回:
          - 所有 block 的中间输出列表（仍然是 C_in 通道）
          - 最终 (降通道 + 上采样后) 的 x
        """
        outputs = []
        for blk in self.blocks:
            x = blk(x)
            outputs.append(x)

        x = self.channel_reduce(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return outputs, x


# ================= DeConvNeXt for DINOv3 =================
class DeConvNeXtDINOv3Base(nn.Module):
    """
    相对对称于 DINOv3 ConvNeXt-Base 的解码器。
    - 保持空间尺度：7x7 -> 14x14 -> 28x28 -> 56x56 -> 224x224
    - 相比旧版：每个 stage block 数量更多，且引入 Conv / Transformer 混合结构
    - dual_stream=True 时：所有通道翻倍（dino + donut）
    输入：(B, 1024, 7, 7) 或 (B, 2048, 7, 7)
    输出：(B, 3, 224, 224) 或 (B, 6, 224, 224)
    """

    def __init__(
        self,
        dual_stream: bool = False,
        output_norm: bool = False,
        out_channels: int = 3,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dual_stream = dual_stream

        # ===== 对称通道配置（单流） =====
        base_inplanes = 512
        base_dims = [128, 64]

        if dual_stream:
            # 所有通道翻倍
            inplanes = base_inplanes * 2
            dims = [d * 2 for d in base_dims]
            out_ch = out_channels * 2
        else:
            inplanes = base_inplanes
            dims = base_dims
            out_ch = out_channels

        # ===== 每个 stage 的 block 结构设计 =====
        # 设计原则：
        #   - 深层（stage3）：Transformer 比例高一点，语义对齐强
        #   - 中层（stage2）：ConvNeXt + Conv + 少量 Transformer，兼顾结构与纹理
        #   - 浅层（stage1）：主要 Conv / ConvNeXt，负责细节与纹理

        stage2_blocks = [
            "transformer",#12
            "transformer",#11
            "transformer",#10
            "transformer",#9
            "convnext",#8
            "convnext",#7
            "conv",#6
            "conv",#5
        ]  # 4 层
        stage1_blocks = [
            "convnext",#4
            "convnext",#3
            "convnext",#2
            "convnext",#1
        ]  # 4 层（原来是 3 个 convnext）

        # ===== 对称结构（scale 一致：14→28→56） =====

        self.re_stage2 = ReStage(
            in_channels=inplanes,
            out_channels=dims[0],
            block_types=stage2_blocks,
            layer_scale_init_value=layer_scale_init_value,
            upsample=True,
        )
        self.re_stage1 = ReStage(
            in_channels=dims[0],
            out_channels=dims[1],
            block_types=stage1_blocks,
            layer_scale_init_value=layer_scale_init_value,
            upsample=True,
        )

        # 最后一段: 56x56 -> 224x224 (×4)，改为双线性插值 + Conv
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[1], dims[1] // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dims[1] // 2, out_ch, kernel_size=3, padding=1),
        )

        self.output_norm = output_norm

    def forward(
        self,
        x_dino: torch.Tensor,
        x_donut: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        """
        返回:
          all_layers: List[Tensor]
            - 所有中间 block 输出（从深到浅）
            - 最终重建图像（append 在最后）
        """
        if self.dual_stream:
            assert x_donut is not None, "dual_stream=True 时必须提供 x_donut"
            x = torch.cat([x_dino, x_donut], dim=1)
        else:
            x = x_dino

        all_layers: List[torch.Tensor] = []

        # stage2: 14x14 -> 28x28
        f2, x = self.re_stage2(x)
        all_layers.extend(f2)

        # stage1: 28x28 -> 56x56
        f1, x = self.re_stage1(x)
        all_layers.extend(f1)

        # output: 56x56 -> 224x224
        out = self.final_upsample(x)
        all_layers.append(out)

        if self.output_norm:
            all_layers = [F.normalize(f, dim=1) for f in all_layers]

        return all_layers


def de_convnext_dinov3_base(**kwargs: Any) -> DeConvNeXtDINOv3Base:
    return DeConvNeXtDINOv3Base(**kwargs)


# ================= Test =================
if __name__ == "__main__":
    print("Single stream:")
    m = de_convnext_dinov3_base()
    x = torch.randn(1, 1024, 7, 7)
    layers = m(x)
    print(f"Total layers: {len(layers)}")
    for i, v in enumerate(layers):
        print(f"{i:02d}: {tuple(v.shape)}")

    print("\nDual stream:")
    m2 = de_convnext_dinov3_base(dual_stream=True)
    x1, x2 = torch.randn(1, 1024, 7, 7), torch.randn(1, 1024, 7, 7)
    layers = m2(x1, x2)
    print(f"Total layers: {len(layers)}")
    for i, v in enumerate(layers):
        print(f"{i:02d}: {tuple(v.shape)}")
