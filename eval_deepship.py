"""
Evaluation script for HTS-AT on DeepShip dataset.
Loads a trained checkpoint and evaluates on the validation set.

Usage:
    python eval_deepship.py /path/to/checkpoint.ckpt
    python eval_deepship.py ./workspace/results/exp_htsat_deepship/checkpoint/l-XX-0.XXX.ckpt
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Deepship_config as config
from data_generator import DeepShip_Dataset
from model.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper

CLASS_NAMES = ['Cargo', 'Passengership', 'Tanker', 'Tug']


def evaluate(checkpoint_path, batch_size=16, num_workers=4):
    # Note: config.classes_num must match what was used during training.
    # Current config has 50 (ESC-50 default). For DeepShip with 4 classes,
    # the config should be updated if you retrain:
    #   config.classes_num = 4
    config.dataset_type = "deepship"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # build model
    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config=config,
        depths=config.htsat_depth,
        embed_dim=config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )

    # load checkpoint
    model = SEDWrapper.load_from_checkpoint(
        checkpoint_path,
        sed_model=sed_model,
        config=config,
        dataset=None,
        strict=False
    )
    model = model.to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # eval dataset
    eval_dataset = DeepShip_Dataset(config=config, eval_mode=True)
    eval_loader = DataLoader(
        dataset=eval_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False
    )
    print(f"Eval samples: {len(eval_dataset)}")

    # inference
    all_preds, all_targets, all_names = [], [], []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            waveform = batch["waveform"].to(device)
            output_dict = model.sed_model(waveform, None, True)
            pred = output_dict["clipwise_output"].cpu().numpy()

            all_preds.append(pred)
            all_targets.append(batch["target"].numpy())
            all_names.extend(batch["audio_name"])

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    pred_labels = np.argmax(preds, axis=1)

    # overall accuracy
    acc = np.mean(pred_labels == targets)
    print(f"\n{'='*50}")
    print(f"  Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}")

    # per-class accuracy
    for i in range(len(CLASS_NAMES)):
        mask = targets == i
        n = mask.sum()
        if n > 0:
            ca = np.mean(pred_labels[mask] == targets[mask])
            print(f"  {CLASS_NAMES[i]:<15}: {ca:.4f} ({ca*100:.2f}%)   [{n} samples]")

    # confusion matrix (only for classes present in data)
    n_classes = len(CLASS_NAMES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(targets, pred_labels):
        if t < n_classes:
            cm[int(t)][int(p) if int(p) < n_classes else 0] += 1

    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    header = "               " + "".join(f"{c:<8}" for c in CLASS_NAMES)
    print(f"  {header}")
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:<12} " + "".join(f"{v:<8}" for v in row))
    print()

    return acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate HTS-AT on DeepShip')
    parser.add_argument('checkpoint', type=str, default='/data/zcx/wav_prj/HTS-Audio-Transformer/workspace/results/exp_htsat_deepship/checkpoint/lightning_logs/version_0/checkpoints/l-epoch=0-acc=0.591.ckpt', help='Path to .ckpt checkpoint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # 也可以直接在这里硬编码路径，例如：
    # args.checkpoint = "./workspace/results/exp_htsat_deepship/checkpoint/l-0-0.615.ckpt"

    evaluate(args.checkpoint, args.batch_size, args.num_workers)
