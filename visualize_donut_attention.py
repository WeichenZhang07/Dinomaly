import os
import argparse
import glob
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision.transforms.functional import to_tensor
from transformers import DonutProcessor, VisionEncoderDecoderModel


def token_features(tokens: torch.Tensor) -> np.ndarray:
    """Compute per-token statistics used for clustering."""
    # tokens: (1, N, C)
    energy = tokens.pow(2).sum(dim=-1).sqrt()
    variance = tokens.var(dim=-1, unbiased=False)
    mean_abs = tokens.abs().mean(dim=-1)
    feats = torch.stack([energy, variance, mean_abs], dim=-1)
    return feats.squeeze(0).cpu().numpy()


def choose_text_cluster(features: np.ndarray, labels: np.ndarray) -> int:
    metrics = []
    for cluster_id in np.unique(labels):
        idx = labels == cluster_id
        cluster_stats = features[idx].mean(axis=0)
        score = cluster_stats[0] + cluster_stats[1]
        metrics.append((score, cluster_id))
    metrics.sort(reverse=True)
    return metrics[0][1]


def save_mask(mask: np.ndarray, layer_idx: int, img_basename: str, out_dir: str):
    mask_img = (mask.astype(np.uint8) * 255)
    pil_mask = Image.fromarray(mask_img)
    out_path = os.path.join(out_dir, f"{img_basename}_layer{layer_idx}_mask.png")
    pil_mask.save(out_path)
    print(f"Saved mask: {out_path}")


# =================================================================================
# Main Execution
# =================================================================================

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Load Model and Processor ---
    print("Loading Donut model and processor...")
    # These will be loaded from the local cache if previously downloaded.
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base", local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", local_files_only=True)
    encoder = model.encoder
    
    # --- Prepare directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(args.input_dir, '*.[jJ][pP][gG]')) + \
                  glob.glob(os.path.join(args.input_dir, '*.[pP][nN][gG]'))
    
    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    # --- Process each image ---
    for img_path in image_paths:
        print(f"\nProcessing image: {img_path}")
        original_img = Image.open(img_path).convert('RGB')
        img_tensor = to_tensor(original_img)

        model.to(device)
        model.eval()
        processor_image = processor(original_img, return_tensors="pt")
        pixel_values = processor_image.pixel_values.to(device)

        with torch.no_grad():
            outputs = model.encoder(pixel_values, output_attentions=False, output_hidden_states=True)

        hidden_states = outputs.hidden_states[1:]  # skip patch embedding output
        base_h = model.encoder.config.image_size[0] // model.encoder.config.patch_size
        base_w = model.encoder.config.image_size[1] // model.encoder.config.patch_size
        aspect = base_h / base_w

        # Iterate over each layer's hidden states and process them individually
        for layer_idx, tokens in enumerate(hidden_states):
            tokens = tokens.detach().to(device)
            feats = token_features(tokens)
            num_tokens = feats.shape[0]

            # Reshape to fit image dimensions based on aspect ratio
            H = int(round((num_tokens * aspect) ** 0.5))
            if H == 0:
                raise ValueError("Failed to infer spatial height from tokens")
            W = num_tokens // H
            if H * W != num_tokens:
                W = int(round((num_tokens / aspect) ** 0.5))
                H = num_tokens // W if W != 0 else 0
            if H * W != num_tokens:
                raise ValueError(
                    f"Token count {num_tokens} cannot be reshaped using inferred aspect (H={H}, W={W})"
                )

            # Apply K-means clustering to the features
            km = KMeans(n_clusters=2, random_state=0, n_init=10)
            labels = km.fit_predict(feats)
            text_cluster = choose_text_cluster(feats, labels)
            mask = (labels == text_cluster).astype(np.uint8).reshape(H, W)

            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            save_mask(mask, layer_idx, img_basename, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster Donut encoder tokens per layer using KMeans.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save token masks.')
    
    # Set default offline mode for HuggingFace
    os.environ["HF_HUB_OFFLINE"] = "1"
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ.pop(k, None)

    args = parser.parse_args()
    main(args)
