import os
import torch
# Disable xFormers on CPU to avoid unsupported attention ops
if not torch.cuda.is_available():
    os.environ["XFORMERS_DISABLED"] = "1"
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from functools import partial
from dataset import MVTecAD2Dataset, DatasetSplit
from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2


# ============ 工具函数 ============
def cosine_similarity_map(f1, f2):
    """计算像素级余弦相似度"""
    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)
    return (f1 * f2).sum(dim=1, keepdim=True)  # (B,1,H,W)


def overlay_heatmap_on_image(image, sim_map):
    """叠加蓝-红色相似度热图到原图。

    - 自动将相似度图插值到与原图相同的分辨率
    - 使用绝对值表示强弱，coolwarm 颜色映射
    """
    H, W = image.shape[-2:]
    sim_t = sim_map
    if isinstance(sim_t, np.ndarray):
        sim = sim_t
    else:
        # sim_map 期望形状 (B,1,h,w) 或 (1,h,w)
        if sim_t.dim() == 3:
            sim_t = sim_t.unsqueeze(0)
        if sim_t.shape[-2:] != (H, W):
            sim_t = F.interpolate(sim_t, size=(H, W), mode='bilinear', align_corners=False)
        sim = sim_t.squeeze().detach().cpu().numpy()

    sim = np.clip(sim, -1, 1)
    sim = np.abs(sim)
    cmap = plt.get_cmap('coolwarm')
    heat = cmap(sim)[:, :, :3]

    img_np = image.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    overlay = 0.6 * img_np + 0.4 * heat.astype(np.float32)
    return torch.from_numpy(overlay).permute(2, 0, 1)


def concat_panel(image, sims):
    """拼接原图 + 每层相似图 + 最终平均图"""
    imgs = [image] + [overlay_heatmap_on_image(image, s) for s in sims]
    return torch.cat(imgs, dim=2)  # 横向拼接


<<<<<<< HEAD
def scores_heatmap_image(image, sim_map, cmap_name: str = 'coolwarm') -> torch.Tensor:
    """仅显示分数热图（不叠加原图）。返回张量 CHW、范围[0,1]。

    - 会将 sim_map 插值到与 image 相同分辨率
    - 使用 abs(cosine) 作为强度
    """
    H, W = image.shape[-2:]
    sim_t = sim_map
    if isinstance(sim_t, np.ndarray):
        sim = sim_t
        if sim.shape != (H, W):
            sim = torch.from_numpy(sim).unsqueeze(0).unsqueeze(0).float()
            sim = F.interpolate(sim, size=(H, W), mode='bilinear', align_corners=False).squeeze().numpy()
    else:
        if sim_t.dim() == 3:
            sim_t = sim_t.unsqueeze(0)
        if sim_t.shape[-2:] != (H, W):
            sim_t = F.interpolate(sim_t, size=(H, W), mode='bilinear', align_corners=False)
        sim = sim_t.squeeze().detach().cpu().numpy()

    sim = np.clip(sim, -1, 1)
    sim = np.abs(sim)
    cmap = plt.get_cmap(cmap_name)
    heat = cmap(sim)[:, :, :3].astype(np.float32)  # H,W,3 in [0,1]
    return torch.from_numpy(heat).permute(2, 0, 1)


=======
>>>>>>> 95543f4ba2cde27ac98c669cec890da83a8ccb07
def _strip_module_prefix(state_dict: dict) -> dict:
    """移除由于DataParallel/DDP导致的 'module.' 前缀。"""
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(isinstance(k, str) and k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_into_model(model: nn.Module, ckpt_path: str, device: str = "cpu") -> None:
    """稳健地将多种格式的权重加载到模型。

    支持：
    - 直接保存的 state_dict
    - 带有 'model_state_dict' 键的字典
    兼容 PyTorch 2.6 的安全加载（weights_only），必要时放行常见 numpy 类型，
    并在信任文件前提下回退到 weights_only=False。
    """
    print(f"[INFO] 加载权重文件: {ckpt_path}")
    state = None

    # 1) 优先尝试安全加载
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            state = ckpt
    except Exception as e:
        print(f"[WARN] 安全加载失败: {type(e).__name__}: {e}")

    # 2) 放行 numpy 常见类型后重试安全加载
    if state is None:
        try:
            import numpy as _np
            torch.serialization.add_safe_globals([
                _np.dtype,
                _np.ufunc,
                _np.core.multiarray.scalar,
            ])
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict):
                state = ckpt
            else:
                state = ckpt
            print("[INFO] 已通过 numpy 允许列表重试加载。")
        except Exception as e:
            print(f"[WARN] 允许列表方式仍失败: {type(e).__name__}: {e}")

    # 3) 最终回退：完全信任的加载（仅在你信任该文件时使用）
    if state is None:
        print("[WARN] 回退到 weights_only=False（仅在你信任该权重文件时使用）")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise RuntimeError("不支持的权重格式：期望为字典/OrderedDict。")

    # 规范化键并加载
    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] 缺失键 {len(missing)} 个（最多显示10个）: {missing[:10]}")
    if unexpected:
        print(f"[WARN] 多余键 {len(unexpected)} 个（最多显示10个）: {unexpected[:10]}")
    print("[INFO] 权重已加载到模型。")


# ============ 主函数 ============
@torch.inference_mode()
def visualize_ad2(model, dataset, device='cuda:0', save_dir='./vis_results', max_items=None):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"[INFO] 共 {len(loader)} 张测试图像，将保存到 {save_dir}")

    for i, item in enumerate(tqdm(loader)):
        if max_items and i >= max_items:
            break

        img = item['image'].to(device)
        img_path = item['image_path'][0]
        cls = item.get('classname', ['unknown'])[0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        cls_dir = os.path.join(save_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        # 获取每层 encoder-decoder 对齐特征
        en_feats, de_feats = model(img)
        sims = []
        for e, d in zip(en_feats, de_feats):
            if e.shape[-2:] != d.shape[-2:]:
                d = F.interpolate(d, size=e.shape[-2:], mode='bilinear', align_corners=False)
            sims.append(cosine_similarity_map(e, d))

        # 最终融合：所有层平均
        final_sim = torch.mean(torch.stack(sims, dim=0), dim=0)
        sims.append(final_sim)

<<<<<<< HEAD
    # 拼接成整张图（原图+各层叠加+最终平均叠加）
    panel = concat_panel(img[0].cpu(), sims)
    save_path = os.path.join(cls_dir, f"{img_name}.png")
    save_image(panel, save_path)

    # 另存一张“纯分数图”（不含背景），使用最终平均热图
    final_sim_map = sims[-1]  # (B,1,h,w)
    score_rgb = scores_heatmap_image(img[0].cpu(), final_sim_map)
    score_save = os.path.join(cls_dir, f"{img_name}_scores.png")
    # save_image expects [0,1]
    save_image(score_rgb, score_save)
=======
        # 拼接成整张图
        panel = concat_panel(img[0].cpu(), sims)
        save_path = os.path.join(cls_dir, f"{img_name}.png")
        save_image(panel, save_path)
>>>>>>> 95543f4ba2cde27ac98c669cec890da83a8ccb07

    print(f"[DONE] 所有结果已保存至 {save_dir}")


# ============ 示例调用 ============
if __name__ == "__main__":
    # ---- 配置 ----
    data_root = "/root/autodl-tmp/mvtec_ad_2"
    classname = "can"
    pth_path = "./saved_results/mvtec2_uni_dinov2/model_iter_1000.pth"
    save_root = "./vis_results_ad2"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- 加载数据 ----
    test_ds = MVTecAD2Dataset(
        source=data_root,
        classname=classname,
        imagesize=1022,
        split=DatasetSplit.TEST,
        preserve_aspect_ratio=False,
        resize_strategy='short_side',
        center_crop=False,
    )

    # ---- 构建模型 ----
    encoder = vit_encoder.load("dinov2reg_vit_base_14")
    embed_dim, num_heads = 768, 12
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

    bottleneck = nn.ModuleList([
        bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)
    ])
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                 attn=LinearAttention2)
        for _ in range(8)
    ])

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
    ).to(device)

    # ---- 加载训练权重 ----
    load_checkpoint_into_model(model, pth_path, device=device)
    print(f"[INFO] 已加载预训练权重: {pth_path}")

    # ---- 可视化 ----
    visualize_ad2(model, test_ds, device=device, save_dir=save_root, max_items=30)
