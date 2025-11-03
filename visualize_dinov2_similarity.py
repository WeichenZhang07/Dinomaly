import argparse
import os
import glob
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless save
import matplotlib.pyplot as plt

from dataset import get_data_transforms


def is_image_file(p: str) -> bool:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    return os.path.splitext(p)[1].lower() in exts


def expand_errors_path(errors: str) -> List[str]:
    # Accept: directory, text file (one path per line), single image, or glob pattern
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
        # Single file
        return [errors]
    # glob pattern
    paths = glob.glob(errors)
    return [p for p in paths if is_image_file(p)]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def round_to_multiple(x: int, m: int) -> int:
    # round to nearest multiple of m, with minimum m
    r = int(round(x / m) * m)
    return max(m, r)


def preprocess_resize_to_patch_multiple(img: Image.Image, patch_hw: Tuple[int, int],
                                        target_size: Tuple[int, int] = None,
                                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Resize image to a rectangle whose width/height are multiples of patch size, optionally to a provided target_size.
    Returns: tensor [1,3,H,W], resized RGB image as uint8 numpy array for overlay, and (W, H).
    """
    if target_size is None:
        w, h = img.size
        pw, ph = patch_hw[1], patch_hw[0]
        Wt = round_to_multiple(w, pw)
        Ht = round_to_multiple(h, ph)
    else:
        Wt, Ht = target_size

    img_resized = img.resize((Wt, Ht), resample=Image.BICUBIC)
    np_img = np.array(img_resized).astype(np.float32) / 255.0  # H, W, 3 in [0,1]
    # Normalize
    x = np_img.copy()
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    x = (x - mean[None, None, :]) / std[None, None, :]
    x = np.transpose(x, (2, 0, 1))  # 3,H,W
    tensor = torch.from_numpy(x).unsqueeze(0)  # 1,3,H,W
    return tensor, (np_img * 255.0).astype(np.uint8), (Wt, Ht)


def extract_layer_tokens(encoder, x: torch.Tensor, target_layers: List[int]) -> List[torch.Tensor]:
    """
    Returns a list of tensors (one per layer index in target_layers), each with shape [1, N_tokens, C].
    This mirrors the extraction pattern used in models.uad.ViTill.
    """
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


def tokens_to_patch_grid(x: torch.Tensor, num_register_tokens: int, h_patches: int, w_patches: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Convert tokens [1, N_tokens, C] to patch tokens [1, HW, C] and return (tokens, side).
    CLS token and register tokens are removed.
    """
    assert x.ndim == 3 and x.shape[0] == 1
    # drop CLS + register tokens
    x_no_cls = x[:, 1 + num_register_tokens:, :]
    hw = x_no_cls.shape[1]
    assert h_patches * w_patches == hw, f"Token grid mismatch: got {hw}, expected {h_patches}x{w_patches}"
    return x_no_cls, (h_patches, w_patches)


def cosine_map_per_patch(ref_tokens: torch.Tensor, err_tokens: torch.Tensor) -> torch.Tensor:
    """
    ref_tokens, err_tokens: [1, HW, C]
    Return: [H, W] similarity map in [0,1] (absolute cosine similarity)
    """
    ref_norm = F.normalize(ref_tokens, dim=-1)
    err_norm = F.normalize(err_tokens, dim=-1)
    sim = (ref_norm * err_norm).sum(dim=-1)  # [1, HW]
    sim = sim.squeeze(0)  # [HW]
    sim_abs = sim.abs()  # [0,1]
    return sim_abs


def overlay_heatmap_on_image(sim_map: np.ndarray, base_rgb_uint8: np.ndarray, alpha: float = 0.5,
                             cmap_name: str = 'coolwarm') -> np.ndarray:
    """
    sim_map: [H, W] in [0,1]
    base_rgb_uint8: [H, W, 3] uint8
    Returns blended image uint8
    """
    import matplotlib.cm as cm
    h, w = sim_map.shape
    # colormap to RGB [0,1]
    cmap = cm.get_cmap(cmap_name)
    cm_rgb = cmap(sim_map)[..., :3]  # drop alpha
    base = base_rgb_uint8.astype(np.float32) / 255.0
    out = (1 - alpha) * base + alpha * cm_rgb
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def plot_layers_heatmaps(sim_maps: List[np.ndarray], layers: List[int], out_path: str, cmap: str = 'coolwarm'):
    n = len(sim_maps)
    # Layout: Try to keep rows small
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
    # hide any extra axes
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis('off')
    fig.tight_layout()
    # add a single colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label('abs(cosine similarity)')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def save_single_layer_maps(sim_maps: List[np.ndarray], layers: List[int], base_out: str, cmap: str = 'coolwarm'):
    vmin, vmax = 0.0, 1.0
    for sim, lidx in zip(sim_maps, layers):
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(sim, vmin=vmin, vmax=vmax, cmap=cmap, origin='upper')
        plt.axis('off')
        out_path = f"{base_out}_layer{lidx:02d}.png"
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Layer-wise absolute cosine similarity visualization (dinov2).')
    parser.add_argument('--ref', type=str, required=True, help='Path to reference (normal) image')
    parser.add_argument('--errors', type=str, required=True, help='Path to error images (dir, glob, txt, or single)')
    parser.add_argument('--output_dir', type=str, default='./similarity_outputs', help='Directory to save results')
    parser.add_argument('--encoder_name', type=str, default='dinov2reg_vit_base_14', help='Encoder name as used in repo')
    parser.add_argument('--layers', type=str, default='all', help='Comma-separated block indices or "all"')
    parser.add_argument('--device', type=str, default=None, help='cuda:0 or cpu; default auto')
    parser.add_argument('--save_individual', action='store_true', help='Also save each layer heatmap separately')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay alpha for heatmap')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Disable xFormers on CPU to avoid unsupported ops
    if str(device).startswith('cpu') or not torch.cuda.is_available():
        os.environ["XFORMERS_DISABLED"] = "1"

    # Import after environment is set
    from models import vit_encoder

    # Load encoder (same path used by training scripts)
    encoder = vit_encoder.load(args.encoder_name)
    encoder.eval().to(device)

    # Determine num_register_tokens for token slicing
    num_register_tokens = getattr(encoder, 'num_register_tokens', 0)
    # Determine patch size
    patch_h, patch_w = encoder.patch_embed.patch_size

    # Parse layers
    if args.layers.strip().lower() == 'all':
        total_layers = len(encoder.blocks)
        target_layers = list(range(total_layers))
    else:
        target_layers = [int(x) for x in args.layers.split(',') if x.strip() != '']

    # Load reference image tensor
    ref_img = load_image(args.ref)
    # Compute target size from reference image (multiples of patch)
    ref_tensor_cpu, ref_rgb_uint8, (Wt, Ht) = preprocess_resize_to_patch_multiple(ref_img, (patch_h, patch_w))
    ref_tensor = ref_tensor_cpu.to(device)
    h_patches = Ht // patch_h
    w_patches = Wt // patch_w

    # Extract reference per-layer tokens
    ref_feats = extract_layer_tokens(encoder, ref_tensor, target_layers)
    ref_tokens = []
    for ft in ref_feats:
        toks, _ = tokens_to_patch_grid(ft, num_register_tokens, h_patches, w_patches)
        ref_tokens.append(toks)

    # Errors list
    error_paths = expand_errors_path(args.errors)
    if len(error_paths) == 0:
        raise RuntimeError(f"No error images found from: {args.errors}")

    for err_path in error_paths:
        try:
            err_img = load_image(err_path)
            # Resize to the SAME target size as reference for valid patch-wise comparison
            err_tensor_cpu, err_rgb_uint8, _ = preprocess_resize_to_patch_multiple(err_img, (patch_h, patch_w), (Wt, Ht))
            err_tensor = err_tensor_cpu.to(device)

            err_feats = extract_layer_tokens(encoder, err_tensor, target_layers)
            sim_maps = []
            for (rf, ef) in zip(ref_tokens, err_feats):
                etoks, _ = tokens_to_patch_grid(ef, num_register_tokens, h_patches, w_patches)
                sim = cosine_map_per_patch(rf, etoks).detach().cpu().numpy().reshape(h_patches, w_patches)
                sim_maps.append(sim)

            # Save combined grid (heatmaps only)
            base = os.path.splitext(os.path.basename(err_path))[0]
            out_combined = os.path.join(args.output_dir, f"{base}_layers.png")
            plot_layers_heatmaps(sim_maps, target_layers, out_combined, cmap='coolwarm')

            # Save per-layer overlay masks on the resized input image
            for sim, lidx in zip(sim_maps, target_layers):
                # upscale sim to resized dims
                sim_up = torch.from_numpy(sim).unsqueeze(0).unsqueeze(0).float()
                sim_up = F.interpolate(sim_up, size=(Ht, Wt), mode='bilinear', align_corners=False).squeeze().numpy()
                overlay = overlay_heatmap_on_image(sim_up, err_rgb_uint8, alpha=args.alpha, cmap_name='coolwarm')
                out_overlay = os.path.join(args.output_dir, f"{base}_layer{lidx:02d}_overlay.png")
                Image.fromarray(overlay).save(out_overlay)

            # Also save a .npz with raw data for programmatic use
            npz_path = os.path.join(args.output_dir, f"{base}_simmaps.npz")
            np.savez_compressed(npz_path, layers=np.array(target_layers), maps=np.array(sim_maps, dtype=np.float32))
        except Exception as e:
            print(f"[WARN] Failed on {err_path}: {e}")


if __name__ == '__main__':
    main()
