import argparse
import os
import glob
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import get_data_transforms


def is_image_file(p: str) -> bool:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    return os.path.splitext(p)[1].lower() in exts


def expand_errors_path(errors: str) -> List[str]:
    if os.path.isdir(errors):
        paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp'):
            paths.extend(glob.glob(os.path.join(errors, '**', ext), recursive=True))
        return sorted(paths)
    if os.path.isfile(errors):
        if errors.lower().endswith('.txt'):
            with open(errors, 'r') as f:
                lines = [ln.strip() for ln in f.readlines()]
            return [ln for ln in lines if ln and is_image_file(ln)]
        return [errors]
    paths = glob.glob(errors)
    return [p for p in paths if is_image_file(p)]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def round_to_multiple(x: int, m: int) -> int:
    r = int(round(x / m) * m)
    return max(m, r)


def preprocess_resize_to_patch_multiple(img: Image.Image, patch_hw: Tuple[int, int],
                                        target_size: Tuple[int, int] = None,
                                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    if target_size is None:
        w, h = img.size
        pw, ph = patch_hw[1], patch_hw[0]
        Wt = round_to_multiple(w, pw)
        Ht = round_to_multiple(h, ph)
    else:
        Wt, Ht = target_size

    img_resized = img.resize((Wt, Ht), resample=Image.BICUBIC)
    np_img = np.array(img_resized).astype(np.float32) / 255.0
    x = np_img.copy()
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    x = (x - mean[None, None, :]) / std[None, None, :]
    x = np.transpose(x, (2, 0, 1))
    tensor = torch.from_numpy(x).unsqueeze(0)
    return tensor, (np_img * 255.0).astype(np.uint8), (Wt, Ht)


def extract_layer_tokens_vit(encoder, x: torch.Tensor, target_layers: List[int]) -> List[torch.Tensor]:
    """For ViT / DINOv2 encoders"""
    with torch.no_grad():
        x = encoder.prepare_tokens(x)
        feats = []
        for i, blk in enumerate(encoder.blocks):
            if i <= max(target_layers):
                x = blk(x)
            else:
                break
            if i in target_layers:
                feats.append(x)
    return feats


def extract_layer_tokens_convnext(encoder, x: torch.Tensor, target_layers: List[int]) -> List[torch.Tensor]:
    """For ConvNeXt (dinov3_convnext_base) using timm features_only interface.

    Assumes the encoder was created with features_only=True and provides a list of
    stage outputs. We index into that list using target_layers.
    """
    with torch.no_grad():
        feats_all = encoder(x)  # List[Tensor], each (B,C,H,W)
    # Guard if target indices exceed available features
    max_idx = len(feats_all) - 1
    sel = [i for i in target_layers if 0 <= i <= max_idx]
    missing = sorted(set(target_layers) - set(sel))
    if missing:
        print(f"[WARN] ConvNeXt: skipping non-existent layer indices: {missing} (total features={len(feats_all)})")
    return [feats_all[i] for i in sel]


def _donut_flat_blocks(encoder) -> List[torch.nn.Module]:
    """Flatten Donut encoder blocks to a simple list (supports Swin/ViT backbones)."""
    enc = encoder.encoder if hasattr(encoder, "encoder") else encoder
    blocks = []
    # Swin-style: enc.layers = [SwinStage,...], each has .blocks (SwinBlock list)
    if hasattr(enc, "layers"):
        for stage in enc.layers:
            if hasattr(stage, "blocks"):
                blocks.extend(list(stage.blocks))
            else:
                blocks.append(stage)
    # ViT-style: enc.layer = [EncoderLayer,...]
    elif hasattr(enc, "layer"):
        for layer in enc.layer:
            # Some ViT impls may have sub-blocks
            if hasattr(layer, "blocks"):
                blocks.extend(list(layer.blocks))
            else:
                blocks.append(layer)
    else:
        # Fallback: all children as sequence
        blocks = list(enc.children())
    return blocks


def extract_layer_tokens_donut(encoder, x: torch.Tensor, target_layers: List[int]) -> List[torch.Tensor]:
    """For Donut encoder (ViT or Swin): hook flattened blocks and collect outputs."""
    feats: List[torch.Tensor] = []
    hooks: dict = {}

    def save_output_hook(name):
        def hook_fn(module, inp, out):
            hooks[name] = out
        return hook_fn

    flat_blocks = _donut_flat_blocks(encoder)
    if len(flat_blocks) == 0:
        raise RuntimeError("Donut encoder has no discoverable blocks to hook.")

    # Filter target indices to available range
    sel_indices = [i for i in target_layers if 0 <= i < len(flat_blocks)]
    missing_indices = sorted(set(target_layers) - set(sel_indices))
    if missing_indices:
        print(f"[WARN] Donut: skipping non-existent layer indices: {missing_indices} (total blocks={len(flat_blocks)})")

    # Register hooks
    handles = []
    for i in sel_indices:
        handles.append(flat_blocks[i].register_forward_hook(save_output_hook(f"blk{i}")))

    with torch.no_grad():
        _ = encoder(x)

    for h in handles:
        h.remove()

    # Collect in requested order (only those that exist)
    for i in sel_indices:
        if f"blk{i}" in hooks:
            feats.append(hooks[f"blk{i}"])
        else:
            print(f"[WARN] Donut: hook for layer {i} did not fire; skipping.")
    return feats


def tokens_to_patch_grid(x: torch.Tensor, num_register_tokens: int, h_patches: int, w_patches: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    x_no_cls = x[:, 1 + num_register_tokens:, :]
    hw = x_no_cls.shape[1]
    assert h_patches * w_patches == hw, f"Token grid mismatch: got {hw}, expected {h_patches}x{w_patches}"
    return x_no_cls, (h_patches, w_patches)


def cosine_map_per_patch(ref_tokens: torch.Tensor, err_tokens: torch.Tensor) -> torch.Tensor:
    ref_norm = F.normalize(ref_tokens, dim=-1)
    err_norm = F.normalize(err_tokens, dim=-1)
    sim = (ref_norm * err_norm).sum(dim=-1)
    sim_abs = sim.abs()
    return sim_abs


def overlay_heatmap_on_image(sim_map: np.ndarray, base_rgb_uint8: np.ndarray, alpha: float = 0.5,
                             cmap_name: str = 'coolwarm') -> np.ndarray:
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
    cm_rgb = cmap(sim_map)[..., :3]
    base = base_rgb_uint8.astype(np.float32) / 255.0
    out = (1 - alpha) * base + alpha * cm_rgb
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _first_tensor(obj):
    """Return the first torch.Tensor found within obj (tensor/tuple/list/dict), else obj if it's tensor, else None."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for it in obj:
            t = _first_tensor(it)
            if torch.is_tensor(t):
                return t
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            t = _first_tensor(v)
            if torch.is_tensor(t):
                return t
        return None
    return None


def _infer_hw_from_n(n: int, Ht: int, Wt: int) -> Tuple[int, int]:
    """Infer an HxW grid from token count n using image aspect ratio as guidance."""
    import math
    target_ratio = Ht / max(1, Wt)
    best = (int(math.sqrt(n)), int(math.sqrt(n)))
    best_err = float('inf')
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h == 0:
            w = n // h
            ratio = h / max(1, w)
            err = abs(ratio - target_ratio)
            if err < best_err:
                best_err = err
                best = (h, w)
    return best


def feats_to_tokens_and_grid(feats: List[torch.Tensor], default_grid: Tuple[int, int] | None,
                             Ht: int, Wt: int) -> List[Tuple[torch.Tensor, Tuple[int, int] | None]]:
    """Normalize layer outputs to (tokens[B,N,C], grid_hw or None) pairs.

    - If feat is 4D (B,C,H,W): return tokens and (H,W)
    - If feat is 3D (B,N,C): return tokens and default_grid (may be None)
    - If feat is tuple/list/dict: use the first tensor found within
    """
    out = []
    for f in feats:
        t = _first_tensor(f)
        if t is None:
            # fall back to original if it is a tensor already
            t = f if torch.is_tensor(f) else None
        if t is None:
            raise RuntimeError("Layer output does not contain a tensor.")
        if t.ndim == 4:
            B, C, h, w = t.shape
            tok = t.flatten(2).transpose(1, 2)
            out.append((tok, (h, w)))
        elif t.ndim == 3:
            tok = t
            if default_grid is None:
                # try to infer grid from token count and image aspect
                n = tok.shape[1]
                grid = _infer_hw_from_n(n, Ht, Wt)
            else:
                grid = default_grid
            out.append((tok, grid))
        else:
            raise RuntimeError(f"Unsupported feature ndim={t.ndim}")
    return out


def plot_layers_heatmaps(sim_maps: List[np.ndarray], layers: List[int], out_path: str, cmap: str = 'coolwarm'):
    n = len(sim_maps)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), squeeze=False)
    vmin, vmax = 0.0, 1.0
    for idx, sim in enumerate(sim_maps):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        im = ax.imshow(sim, vmin=vmin, vmax=vmax, cmap=cmap, origin='upper')
        ax.set_title(f"Layer {layers[idx]}")
        ax.axis('off')
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis('off')
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label('abs(cosine similarity)')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Layer-wise similarity visualization (DINOv2 / Donut).')
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--errors', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./similarity_outputs')
    parser.add_argument('--encoder_type', type=str, default='dinov2', choices=['dinov2', 'donut', 'dinov3_convnext_base'])
    parser.add_argument('--encoder_name', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--layers', type=str, default='all')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--cmap', type=str, default='coolwarm', help='Colormap for score-only visualization')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ========= Load Encoder ==========
    if args.encoder_type == 'dinov2':
        from models import vit_encoder
        encoder = vit_encoder.load(args.encoder_name).eval().to(device)
        patch_h, patch_w = encoder.patch_embed.patch_size
        num_register_tokens = getattr(encoder, 'num_register_tokens', 0)
        extract_fn = extract_layer_tokens_vit

    elif args.encoder_type == 'donut':
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        encoder = model.encoder.eval().to(device)
        patch_h, patch_w = (14, 14)  # approximate ViT patch size
        num_register_tokens = 0
        extract_fn = extract_layer_tokens_donut
    elif args.encoder_type == 'dinov3_convnext_base':
        import timm
        # Use features_only to get intermediate feature maps
        encoder = timm.create_model('convnext_base', pretrained=True, features_only=True).eval().to(device)
        # Determine patch size approximation by final downsampling (convnext_base total stride 32)
        patch_h, patch_w = (32, 32)
        num_register_tokens = 0
        extract_fn = extract_layer_tokens_convnext

    # ========= Parse Layers ==========
    if args.encoder_type == 'dinov2':
        total_layers = len(encoder.blocks) if hasattr(encoder, "blocks") else 12
    elif args.encoder_type == 'donut':
        total_layers = len(_donut_flat_blocks(encoder))
    elif args.encoder_type == 'dinov3_convnext_base':
        # features_only convnext returns a list of 5 feature maps by default (stages outputs)
        # We expose them as layers 0..len-1
        total_layers = len(encoder.feature_info.channels()) if hasattr(encoder, 'feature_info') else 5
    if args.layers.strip().lower() == 'all':
        target_layers = list(range(total_layers))
    else:
        target_layers = [int(x) for x in args.layers.split(',') if x.strip() != '']

    # ========= Reference ==========
    ref_img = load_image(args.ref)
    ref_tensor_cpu, ref_rgb_uint8, (Wt, Ht) = preprocess_resize_to_patch_multiple(ref_img, (patch_h, patch_w))
    ref_tensor = ref_tensor_cpu.to(device)
    h_patches, w_patches = Ht // patch_h, Wt // patch_w
    ref_feats = extract_fn(encoder, ref_tensor, target_layers)
    # For ViT path, default grid is patch grid; for Donut, let it infer or use 4D shapes
    default_grid = (h_patches, w_patches) if args.encoder_type == 'dinov2' else None
    ref_pairs = feats_to_tokens_and_grid(ref_feats, default_grid, Ht, Wt)

    # ========= Error Images ==========
    error_paths = expand_errors_path(args.errors)
    for err_path in error_paths:
        err_img = load_image(err_path)
        err_tensor_cpu, err_rgb_uint8, _ = preprocess_resize_to_patch_multiple(err_img, (patch_h, patch_w), (Wt, Ht))
        err_tensor = err_tensor_cpu.to(device)
        err_feats = extract_fn(encoder, err_tensor, target_layers)
        err_pairs = feats_to_tokens_and_grid(err_feats, default_grid, Ht, Wt)

        sim_maps = []
        for (rf, rgrid), (ef, egrid) in zip(ref_pairs, err_pairs):
            # Ensure both token sequences match length
            if rf.shape[1] != ef.shape[1]:
                # If shapes differ (e.g., due to strides), interpolate to the smaller token length via spatial maps
                # Prefer converting both back to spatial using their grids then aligning
                rH, rW = rgrid if rgrid is not None else _infer_hw_from_n(rf.shape[1], Ht, Wt)
                eH, eW = egrid if egrid is not None else _infer_hw_from_n(ef.shape[1], Ht, Wt)
                # Reshape to (B,C,H,W) by transposing tokens back
                rf_4d = rf.transpose(1, 2).reshape(rf.shape[0], rf.shape[2], rH, rW)
                ef_4d = ef.transpose(1, 2).reshape(ef.shape[0], ef.shape[2], eH, eW)
                # Match to the larger spatial size for better fidelity
                tgtH, tgtW = max(rH, eH), max(rW, eW)
                rf_4d = F.interpolate(rf_4d, size=(tgtH, tgtW), mode='bilinear', align_corners=False)
                ef_4d = F.interpolate(ef_4d, size=(tgtH, tgtW), mode='bilinear', align_corners=False)
                rf = rf_4d.flatten(2).transpose(1, 2)
                ef = ef_4d.flatten(2).transpose(1, 2)
                grid = (tgtH, tgtW)
            else:
                grid = rgrid if rgrid is not None else egrid
                if grid is None:
                    grid = _infer_hw_from_n(rf.shape[1], Ht, Wt)

            sim = cosine_map_per_patch(rf, ef).detach().cpu().numpy().reshape(grid[0], grid[1])
            sim_maps.append(sim)

        base = os.path.splitext(os.path.basename(err_path))[0]
        out_combined = os.path.join(args.output_dir, f"{base}_layers.png")
        plot_layers_heatmaps(sim_maps, target_layers, out_combined)

        for sim, lidx in zip(sim_maps, target_layers):
            sim_up = torch.from_numpy(sim).unsqueeze(0).unsqueeze(0).float()
            sim_up = F.interpolate(sim_up, size=(Ht, Wt), mode='bilinear', align_corners=False).squeeze().numpy()

            # Save overlay (scores on top of image)
            overlay = overlay_heatmap_on_image(sim_up, err_rgb_uint8, alpha=args.alpha, cmap_name=args.cmap)
            out_overlay = os.path.join(args.output_dir, f"{base}_layer{lidx:02d}_overlay.png")
            Image.fromarray(overlay).save(out_overlay)

            # Save standalone score visualization (heatmap only) to avoid background interference
            sim_norm = (sim_up - sim_up.min()) / (sim_up.max() - sim_up.min() + 1e-8)
            cmap = plt.get_cmap(args.cmap)
            sim_rgb = (cmap(sim_norm)[..., :3] * 255.0).astype(np.uint8)
            out_scores = os.path.join(args.output_dir, f"{base}_layer{lidx:02d}_scores.png")
            Image.fromarray(sim_rgb).save(out_scores)


if __name__ == '__main__':
    main()
