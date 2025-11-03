import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from functools import partial

from models.uad import ViTill
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import evaluation_batch, WarmCosineScheduler

from dataset import MVTecAD2Dataset, DatasetSplit, MVTEC_AD2_CLASSNAMES


def round_to_multiple(x: int, m: int) -> int:
    r = int(round(x / m) * m)
    return max(m, r)


def infer_patch_size(encoder_name: str, default: int = 14) -> int:
    try:
        return int(str(encoder_name).split('_')[-1])
    except Exception:
        return default


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    if not logger.handlers:
        logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AD2EvalAdapter(Dataset):
    """Wrap MVTecAD2Dataset to tuple format expected by evaluation_batch.
    Returns (image, mask, label, image_path)
    label: 0 for good, 1 for bad
    """

    def __init__(self, base_ds: MVTecAD2Dataset):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img = item["image"]
        mask = item["mask"]  # [1,H,W]
        is_anomaly = int(item.get("is_anomaly", 0))
        label = torch.tensor([is_anomaly], dtype=torch.long)
        path = item.get("image_path", "")
        return img, mask, label, path


def build_datasets_ad2(data_root, classnames, imagesize=392):
    # Train split per class
    train_datasets = []
    test_adapters = []
    for i, cls in enumerate(classnames):
        train_ds = MVTecAD2Dataset(
            source=data_root,
            classname=cls,
            imagesize=imagesize,
            split=DatasetSplit.TRAIN,
            preserve_aspect_ratio=True,
            resize_strategy='short_side',
            center_crop=True,
        )
        test_ds = MVTecAD2Dataset(
            source=data_root,
            classname=cls,
            imagesize=imagesize,
            split=DatasetSplit.TEST,
            preserve_aspect_ratio=True,
            resize_strategy='short_side',
            center_crop=True,
        )
        train_datasets.append(train_ds)
        test_adapters.append(AD2EvalAdapter(test_ds))
    train_concat = ConcatDataset(train_datasets)
    return train_concat, test_adapters


def train_ad2(args):
    setup_seed(1)

    total_iters = args.total_iters
    batch_size = args.batch_size
    imagesize = args.imagesize  # user-provided target side
    # ensure imagesize is a multiple of ViT patch size
    patch = infer_patch_size(args.encoder_name, default=14)
    if imagesize % patch != 0:
        imagesize_rounded = round_to_multiple(imagesize, patch)
    else:
        imagesize_rounded = imagesize

    # datasets
    # Support both --item_list and --classes (alias). item_list has higher priority if provided.
    item_arg = args.item_list if getattr(args, 'item_list', None) else args.classes
    class_list = item_arg.split(',') if item_arg else MVTEC_AD2_CLASSNAMES
    class_list = [c.strip() for c in class_list if c.strip()]
    # Alias for user familiarity
    item_list = class_list

    train_data, test_adapters = build_datasets_ad2(args.data_path, class_list, imagesize=imagesize_rounded)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # model config
    encoder_name = args.encoder_name  # 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise RuntimeError("Architecture not in small, base, large.")

    bottleneck = nn.ModuleList([
        bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)
    ])

    decoder = []
    for _ in range(8):
        blk = VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2,
        )
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )
    device = args.device
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    # init
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    from optimizers import StableAdamW
    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=args.lr, final_value=args.lr_final,
                                       total_iters=total_iters, warmup_iters=100)

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    print_fn(f"device: {device}")
    print_fn(f"train images: {len(train_data)} | classes: {','.join(class_list)}")
    print_fn(f"item_list: {','.join(item_list)}")
    if imagesize_rounded != imagesize:
        print_fn(f"[note] imagesize adjusted from {imagesize} to {imagesize_rounded} to match patch size {patch}.")

    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_loader)))):
        model.train()
        loss_list = []
        for batch in train_loader:
            img = batch["image"].to(device)

            en, de = model(img)
            # curriculum HM-percent cosine as in original script
            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            from utils import global_cosine_hm_percent
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            lr_scheduler.step()

            loss_list.append(loss.item())

            # periodic evaluation
            if (it + 1) % args.eval_every == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for cls, test_ds in zip(class_list, test_adapters):
                    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
                    results = evaluation_batch(model, test_loader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        f"{cls}: I-Auroc:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}, "
                        f"P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, P-AUPRO:{aupro_px:.4f}")

                print_fn(
                    "Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}".format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                model.train()

            it += 1
            if it >= total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
        if it >= total_iters:
            break


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Dinomaly on MVTec AD 2.0 (unified).')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/mvtec_ad_2', help='Root of mvtec_ad_2')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='mvtec2_uni_dinov2')
    parser.add_argument('--encoder_name', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--classes', type=str, default='can', help='Comma-separated class list; empty=all')
    parser.add_argument('--item_list', type=str, default='', help='Alias of --classes; takes priority if set')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--imagesize', type=int, default=1022)
    parser.add_argument('--total_iters', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_final', type=float, default=2e-4)
    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parse_args()
    train_ad2(args)
