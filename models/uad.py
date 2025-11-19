import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math


class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all


class ViTillCat(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[1, 3, 5, 7],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTillCat, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        for i, blk in enumerate(self.decoder):
            x = blk(x)

        en = [torch.cat([en_list[idx] for idx in self.fuse_layer_encoder], dim=2)]
        de = [x]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

class ViTAD(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 5, 8, 11],
            fuse_layer_encoder=[0, 1, 2],
            fuse_layer_decoder=[2, 5, 8],
            mask_neighbor_size=0,
            remove_class_token=False,
    ) -> None:
        super(ViTAD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
            x = x[:, 1 + self.encoder.num_register_tokens:, :]

        # x = torch.cat(en_list, dim=2)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [en_list[idx] for idx in self.fuse_layer_encoder]
        de = [de_list[idx] for idx in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de


class ViTillv2(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7]
    ) -> None:
        super(ViTillv2, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en.append(x)

        x = self.fuse_feature(en)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de.append(x)

        side = int(math.sqrt(x.shape[1]))

        en = [e[:, self.encoder.num_register_tokens + 1:, :] for e in en]
        de = [d[:, self.encoder.num_register_tokens + 1:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        return en[::-1], de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class ViTillv3(nn.Module):
    def __init__(
            self,
            teacher,
            student,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_dropout=0.,
    ) -> None:
        super(ViTillv3, self).__init__()
        self.teacher = teacher
        self.student = student
        if fuse_dropout > 0:
            self.fuse_dropout = nn.Dropout(fuse_dropout)
        else:
            self.fuse_dropout = nn.Identity()
        self.target_layers = target_layers
        if not hasattr(self.teacher, 'num_register_tokens'):
            self.teacher.num_register_tokens = 0

    def forward(self, x):
        with torch.no_grad():
            patch = self.teacher.prepare_tokens(x)
            x = patch
            en = []
            for i, blk in enumerate(self.teacher.blocks):
                if i <= self.target_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.target_layers:
                    en.append(x)
            en = self.fuse_feature(en, fuse_dropout=False)

        x = patch
        de = []
        for i, blk in enumerate(self.student):
            x = blk(x)
            if i in self.target_layers:
                de.append(x)
        de = self.fuse_feature(de, fuse_dropout=False)

        en = en[:, 1 + self.teacher.num_register_tokens:, :]
        de = de[:, 1 + self.teacher.num_register_tokens:, :]
        side = int(math.sqrt(en.shape[1]))

        en = en.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        de = de.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        return [en.contiguous()], [de.contiguous()]

    def fuse_feature(self, feat_list, fuse_dropout=False):
        if fuse_dropout:
            feat = torch.stack(feat_list, dim=1)
            feat = self.fuse_dropout(feat).mean(dim=1)
            return feat
        else:
            return torch.stack(feat_list, dim=1).mean(dim=1)


class ReContrast(nn.Module):
    def __init__(
            self,
            encoder,
            encoder_freeze,
            bottleneck,
            decoder,
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        en = self.encoder(x)
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x)
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        return en_freeze + en, de

    def train(self, mode=True, encoder_bn_train=True):
        self.training = mode
        if mode is True:
            if encoder_bn_train:
                self.encoder.train(True)
            else:
                self.encoder.train(False)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
        return self


def update_moving_average(ma_model, current_model, momentum=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(old_weight, up_weight)

    for current_buffers, ma_buffers in zip(current_model.buffers(), ma_model.buffers()):
        old_buffer, up_buffer = ma_buffers.data, current_buffers.data
        ma_buffers.data = update_average(old_buffer, up_buffer, momentum)


def update_average(old, new, momentum=0.99):
    if old is None:
        return new
    return old * momentum + (1 - momentum) * new


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Callable, Tuple

# ==== 小工具 ====

def _to_tokens(x: torch.Tensor):
    if isinstance(x, torch.Tensor):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            return x, H, W
        elif x.dim() == 3:
            B, N, C = x.shape
            s = int(N ** 0.5)
            return x, s, s
    raise ValueError(f"Expected torch.Tensor with dim 3/4, got {type(x)}")


def _proj_channels(x: torch.Tensor, out_c: int) -> torch.Tensor:
    """通道投影 (B,N,Cin)->(B,N,Cout)，按 token 维线性变换。"""
    B, N, Cin = x.shape
    if Cin == out_c:
        return x
    proj = nn.Linear(Cin, out_c, bias=False).to(x.device)
    # NOTE: 运行时即时创建 Linear → 为可微但不复用参数（用于“组内一次性对齐”场景）
    return proj(x)


def _resize_tokens(x: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
    """(B,N,C) → reshape成 (B,C,H,W) 双线性插值到 H_out×W_out → 再回到 (B,N,C)。"""
    B, N, C = x.shape
    H = int(math.sqrt(N)); W = H
    feat = x.transpose(1, 2).reshape(B, C, H, W)
    feat = F.interpolate(feat, size=(H_out, W_out), mode="bilinear", align_corners=False)
    out = feat.flatten(2).transpose(1, 2)
    return out


def _fuse(feats: List[torch.Tensor], mode: str = "mean") -> torch.Tensor:
    """组内融合：对齐后 (B,N,C) 列表 → (B,N,C)"""
    if len(feats) == 1:
        return feats[0]
    stack = torch.stack(feats, dim=1)  # (B,K,N,C)
    if mode == "mean":
        return stack.mean(dim=1)
    elif mode == "sum":
        return stack.sum(dim=1)
    else:
        raise ValueError(f"Unsupported fuse mode: {mode}")


# ============ ConvNeXt 特征提取器 ============

def _to_tokens(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """统一输出为 (B, N, C)，返回 H,W"""
    if x.dim() == 4:  # (B,C,H,W)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), H, W
    elif x.dim() == 3:  # (B,N,C)
        B, N, C = x.shape
        s = int(N ** 0.5)
        return x, s, s
    else:
        raise ValueError(f"Unexpected feature shape {x.shape}")


class ConvNeXtExtractor(nn.Module):
    """
    通用 ConvNeXt / DINOv3 ConvNeXt 编码器特征提取器
    支持两种结构来源:
    1) torchvision/timm ConvNeXt: 判断模块是否拥有 dwconv + pwconv2
    2) HuggingFace DINOv3ConvNextModel: stages -> DINOv3ConvNextStage -> DINOv3ConvNextLayer

    捕获策略:
    - 对每个 ConvNeXt block 注册 forward hook
    - 返回特征按正向顺序，统一转为 tokens (B, N, C)
    """

    def __init__(self, model: nn.Module, verbose: bool = False):
        super().__init__()
        self.model = model
        self.verbose = verbose
        self.features: List[torch.Tensor] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if isinstance(output, torch.Tensor):
                # 部分 DINOv3 层输出是 (B, H, W, C)，需转换为 (B, C, H, W)
                if output.dim() == 4 and output.shape[1] not in (1, 3):  # (B,H,W,C)
                    if output.shape[-1] in (64,128,256,512,1024):
                        output = output.permute(0, 3, 1, 2).contiguous()
                self.features.append(output)
                if self.verbose:
                    print(f"[Hook Triggered] {name}: {tuple(output.shape)}")
        return hook

    def _is_hf_dinov3_layer(self, module: nn.Module) -> bool:
        name = module.__class__.__name__
        return (
            "DINOv3ConvNextLayer" in name
            or "DINOv3ConvNextStage" in name
            or ("depthwise_conv" in dir(module) and "pointwise_conv2" in dir(module))
        )

    def _register_hooks(self):
        """递归注册所有 ConvNeXt / DINOv3ConvNext Layer"""
        try:
            from torchvision.models.convnext import CNBlock  # type: ignore
            cnblock_cls = CNBlock
        except Exception:
            cnblock_cls = None

        for name, module in self.model.named_modules():
            is_cnblock = cnblock_cls and isinstance(module, cnblock_cls)
            has_convnext_sign = hasattr(module, "dwconv") and hasattr(module, "pwconv2")
            is_hf_dinov3 = self._is_hf_dinov3_layer(module)
            # ✅ 针对 DINOv3 ConvNeXt: 过滤掉 stage 和 norm 层，仅 hook Layer
            if is_cnblock or has_convnext_sign or is_hf_dinov3:
                if "norm" in name.lower() or "downsample" in name.lower():
                    continue
                h = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(h)
                if self.verbose:
                    print(f"[Hook Registered] {name} ({module.__class__.__name__})")

    def forward(self, x):
        self.features.clear()
        try:
            # torchvision / timm
            _ = self.model(x)
        except TypeError:
            # HuggingFace DINOv3
            _ = self.model(pixel_values=x)
        feats = []
        for f in self.features:
            # 统一转为 (B, N, C)
            if f.dim() == 4:
                B, C, H, W = f.shape
                tok = f.flatten(2).transpose(1, 2)
            else:
                tok = f
            feats.append(tok)
        return feats

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()



# ============ Swin Transformer 特征提取器（含Donut支持） ============

class SwinExtractor(nn.Module):
    """
    ✅ 通用 Donut-Swin 编码器特征提取器
    - hook 所有 DonutSwinLayer
    - 捕获每个 block 的输出
    - 返回 [(B, N, C), ...]
    """

    def __init__(self, model: nn.Module, verbose: bool = False):
        super().__init__()
        self.model = model
        self.verbose = verbose
        self.features: List[torch.Tensor] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if isinstance(output, torch.Tensor):
                self.features.append(output)
                if self.verbose:
                    print(f"[Hook Triggered] {name}: {tuple(output.shape)}")
        return hook

    def _register_hooks(self):
        """递归注册所有 DonutSwinLayer"""
        # HuggingFace Donut 内部定义了 DonutSwinLayer 类
        target_names = ("DonutSwinLayer", "SwinLayer")
        for name, module in self.model.named_modules():
            if any(k in module.__class__.__name__ for k in target_names):
                h = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(h)
                if self.verbose:
                    print(f"[Hook Registered] {name} ({module.__class__.__name__})")

    def forward(self, x):
        self.features.clear()
        _ = self.model(x)
        feats = []
        for f in self.features:
            tok, _, _ = _to_tokens(f)
            feats.append(tok)
        return feats

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
# ==== 主体：ViTillDual ====


import math
from typing import List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------

# -----------------------------
# 主体：修正版 ViTillDual（接口名不变）
# -----------------------------
import math
from typing import List, Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTillDual(nn.Module):
    """
    双编码器版 Dinomaly：ConvNeXt (DINO) + Swin (Donut)

    主要改动：
    - bottleneck 输入不再使用 encoder 最后一层，而是使用 target_a / target_b 态选层多层融合方式
    """

    def __init__(
        self,
        encoder_a: nn.Module,
        encoder_b: nn.Module,
        decoder: nn.Module,
        fuse_layer_encoder,
        fuse_layer_decoder,
        encoder_extractors,
        mode="dual",
        # ---- 新增：可配置 bottleneck 输入层 ----
        target_a=None,   # e.g., [22,23,24,25,...]
        target_b=None,   # e.g., [14,15,16,17,...]
        bottleneck_depth: int = 2,
        bottleneck_dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.decoder = decoder
        self.mode = mode

        self.encA_extractor, self.encB_extractor = encoder_extractors

        # supervision grouping
        self.groups_a = fuse_layer_encoder[0]
        self.groups_b = fuse_layer_encoder[1] if mode=="dual" else None
        self.groups_d = fuse_layer_decoder

        # bottleneck config
        self.bottleneck_depth = bottleneck_depth
        self.bottleneck_dropout = bottleneck_dropout

        # ---- NEW: save bottleneck target layers ----
        self.target_a = target_a or [len(self.groups_a[-1])-1]  # default: last group layer
        self.target_b = target_b or ([len(self.groups_b[-1])-1] if mode=="dual" else None)

        self.bottleneck_blocks = None
        self._split_idx = None


    # ----------------- utilities -----------------
    def _init_bottleneck_if_needed(self, dim, device=None):
        if self.bottleneck_blocks is not None:
            return
        
        if device is None:
            device = next(self.parameters()).device   # fallback

        blocks = []
        for _ in range(self.bottleneck_depth):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Dropout(self.bottleneck_dropout),
                )
            )
        self.bottleneck_blocks = nn.ModuleList(blocks).to(device)



    def _to_tokens(self, feat):
        if feat.dim()==4:
            B,C,H,W = feat.shape
            x = feat.flatten(2).transpose(1,2).contiguous()
            return x,H,W
        elif feat.dim()==3:
            N = feat.shape[1]
            s = int(math.sqrt(N))
            return feat,s,s
        raise ValueError


    def _tokens_to_feat(self, tok):
        B,N,C = tok.shape
        s=int(math.sqrt(N))
        return tok.transpose(1,2).reshape(B,C,s,s)


    def _resize_tokens(self, tok, H,W):
        B,N,C = tok.shape
        s=int(math.sqrt(N))
        feat = tok.transpose(1,2).reshape(B,C,s,s)
        feat = F.interpolate(feat,size=(H,W),mode="bilinear",align_corners=False)
        return feat.flatten(2).transpose(1,2)


    def _align_and_fuse_no_proj(self, toks):
        if len(toks)==1: return toks[0]
        Nmin=min(t.shape[1] for t in toks)
        s=int(math.sqrt(Nmin))
        aligned=[t if t.shape[1]==Nmin else self._resize_tokens(t,s,s) for t in toks]
        return torch.stack(aligned,dim=0).mean(0)


    # ----------------- forward -----------------
    def forward(self, x):

        # 1) Encoder feature extraction
        feats_a = self.encA_extractor(x)
        assert len(feats_a)>0

        if self.mode=="dual":
            feats_b = self.encB_extractor(x)
            assert len(feats_b)>0

        # ---------------------------------------------------------
        # 2) NEW: multi-layer fusion for bottleneck input
        # ---------------------------------------------------------
        # ConvNeXt (DINO)
        toks_a = [feats_a[i] for i in self.target_a]
        fused_a = self._align_and_fuse_no_proj(toks_a)

        if self.mode=="dual":
            toks_b = [feats_b[i] for i in self.target_b]
            fused_b = self._align_and_fuse_no_proj(toks_b)

            # concat channels => bottleneck input
            deep_tokens = torch.cat([fused_a, fused_b], dim=-1)
            self._split_idx = fused_a.shape[-1]
        else:
            deep_tokens = fused_a


        # ---------------------------------------------------------
        # 3) bottleneck residual stacking
        # ---------------------------------------------------------
        self._init_bottleneck_if_needed(deep_tokens.shape[-1])
        for blk in self.bottleneck_blocks:
            deep_tokens = deep_tokens + blk(deep_tokens)


        # ---------------------------------------------------------
        # 4) tokens → feature maps → decoder
        # ---------------------------------------------------------
        if self.mode=="dual":
            xA = self._tokens_to_feat(deep_tokens[..., :self._split_idx])
            xB = self._tokens_to_feat(deep_tokens[..., self._split_idx:])
            dec_layers = self.decoder(x_dino=xA, x_donut=xB)
        else:
            xA = self._tokens_to_feat(deep_tokens)
            dec_layers = self.decoder(xA)


        # ---------------------------------------------------------
        # 5) supervision: encoder / decoder alignment
        # ---------------------------------------------------------
        en_fused=[]
        for i,idxs in enumerate(self.groups_a):
            fa = self._align_and_fuse_no_proj([feats_a[j] for j in idxs])
            if self.mode=="dual":
                fb = self._align_and_fuse_no_proj([feats_b[j] for j in self.groups_b[i]])
                fa = torch.cat([fa,fb],dim=-1)
            en_fused.append(self._tokens_to_feat(fa))

        dec_tokens=[self._to_tokens(d)[0] for d in reversed(dec_layers)]
        de_fused=[self._tokens_to_feat(self._align_and_fuse_no_proj([dec_tokens[j] for j in idxs]))
                  for idxs in self.groups_d]

        return en_fused, de_fused

