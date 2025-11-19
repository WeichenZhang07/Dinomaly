import os
# 如果需要自定义 HF 镜像可在外部环境设置 HF_ENDPOINT；此处不在代码里硬编码 token，避免泄露。
# 用户请在运行前:  export HF_TOKEN=xxxx  或者  huggingface-cli login
import math
import random
import logging
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from models.uad import ViTillDual, ConvNeXtExtractor, SwinExtractor
from models.de_convnext import de_convnext_dinov3_base
from utils import evaluation_batch, WarmCosineScheduler

from dataset import MVTecAD2Dataset, DatasetSplit, MVTEC_AD2_CLASSNAMES


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


def safe_torch_save(obj, path, print_fn=print):
    """Write to a temporary file then atomically move into place. Falls back to legacy format.
    Reduces chance of partial/corrupted checkpoints on network or low-space filesystems.
    """
    tmp_path = path + ".tmp"
    try:
        # Legacy serialization is often more robust with some FS backends
        torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            print_fn(f"Checkpoint save failed at {path}: {e}")
            raise


def setup_seed(seed=1):
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
    train_datasets = []
    test_adapters = []
    for cls in classnames:
        train_ds = MVTecAD2Dataset(
            source=data_root,
            classname=cls,
            imagesize=imagesize,
            split=DatasetSplit.TRAIN,
            preserve_aspect_ratio=False,
            resize_strategy='short_side',
            center_crop=False,
        )
        test_ds = MVTecAD2Dataset(
            source=data_root,
            classname=cls,
            imagesize=imagesize,
            split=DatasetSplit.TEST,
            preserve_aspect_ratio=False,
            resize_strategy='short_side',
            center_crop=False,
        )
        train_datasets.append(train_ds)
        test_adapters.append(AD2EvalAdapter(test_ds))
    train_concat = ConcatDataset(train_datasets)
    return train_concat, test_adapters


def split_two_groups(n: int):
    # Split indices 0..n-1 into two nearly-equal groups
    k = n // 2
    return [list(range(0, k)), list(range(k, n))]


def infer_fused_inplanes(encA: nn.Module, encB: nn.Module, imagesize: int, device: str) -> tuple:
    """Run a tiny forward through extractors to infer channels for deepest group fusion.
    Returns (groups_a, groups_b, fused_inplanes)
    """
    # Prepare extractors
    extA = ConvNeXtExtractor(encA).to(device)
    extB = SwinExtractor(encB).to(device)

    with torch.inference_mode():
        x = torch.zeros(1, 3, imagesize, imagesize, device=device)
        feats_a = extA(x)  # list of (B,N,C)
        feats_b = extB(x)

    n_a = len(feats_a)
    n_b = len(feats_b)
    groups_a = split_two_groups(n_a)
    groups_b = split_two_groups(n_b)

    # Deepest groups are the last groups
    def max_channels_in_group(feats, idxs):
        if not idxs:
            return 0
        return max(f.shape[2] for f in (feats[i] for i in idxs))

    Ca = max_channels_in_group(feats_a, groups_a[-1])
    Cb = max_channels_in_group(feats_b, groups_b[-1])
    fused_c = max(Ca, Cb)
    return groups_a, groups_b, int(fused_c)


def train_dual(args):
    setup_seed(1)

    total_iters = args.total_iters
    batch_size = args.batch_size
    imagesize = args.imagesize
    mode = args.mode

    # datasets
    item_arg = args.item_list if getattr(args, 'item_list', None) else args.classes
    class_list = item_arg.split(',') if item_arg else MVTEC_AD2_CLASSNAMES
    class_list = [c.strip() for c in class_list if c.strip()]

    train_data, test_adapters = build_datasets_ad2(args.data_path, class_list, imagesize=imagesize)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    device = args.device

    # encoders: Donut (encoder) + ConvNeXt Base
    encoder_b = None
    if mode == 'dual':
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base",local_files_only=True)  # not used directly but cached
        donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base",local_files_only=True)
        encoder_b = donut_model.encoder.eval().to(device)  # Swin-like

    # Prefer HuggingFace Dinov3 ConvNeXt encoder when available; fallback to timm ConvNeXt otherwise
    from transformers import AutoModel
    import os, glob

    # -------------------------
    # 强制 OFFLINE 模式，防止联网
    # -------------------------
    os.environ["HF_HUB_OFFLINE"] = "1"
    for k in ["HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"]:
        os.environ.pop(k, None)


    # -------------------------
    # 本地 snapshot 查找方法
    # -------------------------
    def _find_local_snapshot(repo_id: str):
        """
        例如输入: "facebook/dinov3-convnext-base-pretrain-lvd1689m"
        自动找到: ~/.cache/huggingface/hub/models--facebook--dinov3-convnext-base-pretrain-lvd1689m/snapshots/<HASH>
        """
        repo_dir = repo_id.replace("/", "--")
        base = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo_dir}")
        snaps = glob.glob(os.path.join(base, "snapshots", "*"))
        return snaps[0] if snaps else None


    hf_repo = "facebook/dinov3-convnext-base-pretrain-lvd1689m"

    local_path = _find_local_snapshot(hf_repo)

    if local_path:
        print(f"✔ Using local HuggingFace snapshot: {local_path}")
        hf_model = AutoModel.from_pretrained(local_path, local_files_only=True)
    else:
        print(f"⚠ No local cache found for {hf_repo}, falling back to timm.")
        import timm
        hf_model = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True)


    # -------------------------
    # 和你原来一致：自动提取 backbone
    # -------------------------
    encoder_candidate = None
    for attr in ("vision_model", "dinov3", "base_model", "encoder", "backbone", "model"):
        if hasattr(hf_model, attr):
            encoder_candidate = getattr(hf_model, attr)
            print(f"✔ Found backbone at attribute: {attr}")
            break

    encoder_a = (encoder_candidate or hf_model).eval().to(device)


    # Build extractors and infer groups dynamically to avoid hard-coded indices
    extA = ConvNeXtExtractor(encoder_a).to(device)
    extB = None
    groups_b = None
    if mode == 'dual':
        extB = SwinExtractor(encoder_b).to(device)
        with torch.inference_mode():
            x_probe = torch.zeros(1, 3, imagesize, imagesize, device=device)
            feats_b = extB(x_probe)
        groups_b = [[5,6,7,8], [9,10,11,12]]#splited stage2

    groups_a = [[8,9,10,11,12,13,14,15,16,17,18,19,20,21], [22,23,24,25,26,27,28,29,30,31,32,33,34,35]]#splited stage2


    # decoder: DeConvNeXt (dual stream). ViTillDual will pass two branches separately.
    decoder = de_convnext_dinov3_base(dual_stream=(mode=='dual'))

    # groups for decoder: only stage3 as target, 2 groups both map to stage index 2
    fuse_layer_decoder = [[5,6,7,8],[9,10,11,12]] #splited stage2
    fuse_layer_encoder = [groups_a]
    if mode == 'dual':
        fuse_layer_encoder.append(groups_b)

    model = ViTillDual(
        encoder_a=encoder_a,
        encoder_b=encoder_b,
        decoder=decoder,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        encoder_extractors = (extA, extB),
        target_a = [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
        target_b = [5,6,7,8,9,10,11,12] if mode=='dual' else None,
        mode=mode,
    ).to(device)

    # Optimizer and scheduler for decoder only (encoders frozen)
    from optimizers import StableAdamW
    # --- TRAINABLE PARAM SELECTION (minimal updated version) ---

    trainable_params = []

    # 1) Decoder 必训练
    trainable_params += list(model.decoder.parameters())

    # 2) Bottleneck 部分 (替代旧 model.bottleneck)
    if getattr(model, "bottleneck_blocks", None) is not None:
        for blk in model.bottleneck_blocks:
            trainable_params += list(blk.parameters())

    # 3) Dual 模式策略：默认不训练 encoder_b，也不再存在 proj/head 分支
    if mode == 'dual':
        print("⚠ Dual mode: encoder_b is frozen, no additional projection heads required.")
    else:
        print("✔ Single mode: only decoder + bottleneck participate in training.")


    optimizer = StableAdamW([{'params': trainable_params}],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=args.lr, final_value=args.lr_final,
                                       total_iters=total_iters, warmup_iters=100)

    # logger
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    print_fn(f"device: {device}")
    print_fn(f"mode: {mode}")
    print_fn(f"train images: {len(train_data)} | classes: {','.join(class_list)}")
    print_fn(f"groups_a: {groups_a}")
    if mode == 'dual':
        print_fn(f"groups_b: {groups_b}")
    #print_fn(f"decoder inplanes (fused C): {inplanes}")

    # checkpoint dir
    ckpt_dir = os.path.join(args.save_dir, args.save_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    it = 0
    for epoch in range(int(np.ceil(total_iters / max(1, len(train_loader))))):
        model.train()
        loss_list = []
        for batch in train_loader:
            img = batch["image"].to(device)

            outputs = model(img)
            
            # New loss calculation based on model output
            en, de = model(img)   # new API output already (B,C,H,W)
            # nothing else needed — model already fused+reshaped


            # Use only 2-group outputs; ViTillDual returns lists per group
            # curriculum HM-percent cosine as in previous script
            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            from utils import global_cosine_hm_percent
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
            optimizer.step()
            lr_scheduler.step()

            loss_list.append(loss.item())

            # periodic evaluation
            if (it + 1) % args.eval_every == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for cls, test_ds in zip(class_list, test_adapters):
                    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
                    # Accumulate AUROC across evaluations by class until both classes appear
                    results = evaluation_batch(model, test_loader, device, max_ratio=0.01, resize_mask=256,
                                               accumulate_auroc_tag=f"mvtec2:{cls}")
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

            # periodic checkpoint saving (decoder + bottleneck only, to keep checkpoints small)
            if (it + 1) % args.save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"model_iter_{it+1}.pth")
                print_fn(f"Saving checkpoint: {ckpt_path}")
                
                # Construct state dict with all trainable parts
                trainable_state_dict = {
                    'decoder': model.decoder.state_dict(),
                    'bottleneck': model.bottleneck.state_dict(),
                    'fusion': model.fusion_layer.state_dict() if hasattr(model, "fusion_layer") else None,
                    'proj_vit': model.proj_vit.state_dict() if hasattr(model, "proj_vit") else None,
                    'proj_swin': model.proj_swin.state_dict() if hasattr(model, "proj_swin") else None,
                    'recon_head': model.recon_head.state_dict() if hasattr(model, "recon_head") else None,
                    'positional_modules': model.rpe.state_dict() if hasattr(model, "rpe") else None,
                }
                trainable_state_dict = {k: v for k, v in trainable_state_dict.items() if v is not None}

                if mode == 'dual':
                    trainable_state_dict['proj_b_state_dict'] = model.proj_b.state_dict()
                    trainable_state_dict['head_b_state_dict'] = model.head_b.state_dict()

                state = {
                    'iter': it + 1,
                    **trainable_state_dict,
                    'groups': {'a': groups_a, 'b': groups_b, 'd': fuse_layer_decoder},
                    'encoders_meta': {'a': type(model.encoder_a).__name__, 'b': type(model.encoder_b).__name__ if model.encoder_b else None},
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_last_iter': getattr(lr_scheduler, 'last_iter', it+1),
                    'mode': mode,
                }
                # Move tensors to CPU to reduce GPU memory pressure during save
                for k, v in state.items():
                    if isinstance(v, dict) and k.endswith('_state_dict'):
                        state[k] = {key: val.cpu() for key, val in v.items()}

                safe_torch_save(state, ckpt_path, print_fn=print_fn)

            it += 1
            if it >= total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list) if loss_list else float('nan')))
        if it >= total_iters:
            break

    # final checkpoint (decoder + bottleneck only)
    final_path = os.path.join(ckpt_dir, 'model_final.pth')
    print_fn(f"Saving final (small) model to {final_path}")
    
    final_trainable_state_dict = {
        'decoder_state_dict': model.decoder.state_dict(),
        'proj_a_state_dict': model.proj_a.state_dict(),
        'bridge_state_dict': model.bridge.state_dict(),
        'head_a_state_dict': model.head_a.state_dict(),
        'bottleneck_state_dict': model.bottleneck.state_dict(),
    }
    if mode == 'dual':
        final_trainable_state_dict['proj_b_state_dict'] = model.proj_b.state_dict()
        final_trainable_state_dict['head_b_state_dict'] = model.head_b.state_dict()

    final_state = {
        'iter': it,
        **final_trainable_state_dict,
        'groups': {'a': groups_a, 'b': groups_b, 'd': fuse_layer_decoder},
        'encoders_meta': {'a': type(model.encoder_a).__name__, 'b': type(model.encoder_b).__name__ if model.encoder_b else None},
        'mode': mode,
    }
    
    for k, v in final_state.items():
        if isinstance(v, dict) and k.endswith('_state_dict'):
            final_state[k] = {key: val.cpu() for key, val in v.items()}
            
    safe_torch_save(final_state, final_path, print_fn=print_fn)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Dinomaly on MVTec AD 2.0 (dual encoders: Donut + ConvNeXt).')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/mvtec_ad_2', help='Root of mvtec_ad_2')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='mvtec2_dual_donut_convnext')
    parser.add_argument('--classes', type=str, default='can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts', help='Comma-separated class list; empty=all')
    parser.add_argument('--item_list', type=str, default='', help='Alias of --classes; takes priority if set')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--imagesize', type=int, default=512)
    parser.add_argument('--total_iters', type=int, default=2000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=500, help='Save checkpoint every N iterations')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_final', type=float, default=2e-4)
    parser.add_argument('--mode', type=str, default='dual', choices=['dual', 'single'], help='Model mode: dual or single encoder')
    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parse_args()
    train_dual(args)
