# import basic packages
import os
import numpy as np
import wget
import sys
import gdown
import zipfile
import librosa
import soundfile as sf
# in the notebook, we only can use one GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Load the model package
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings

from utils import create_folder, dump_config, process_idc
import Deepship_config as config
from sed_model import SEDWrapper, Ensemble_SEDWrapper
from data_generator import ESC_Dataset, DeepShip_Dataset
from model.htsat import HTSAT_Swin_Transformer

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc='Training')

    for batch in pbar:
        input, label = batch['waveform'], batch['target']
        input, label = input.to(device), label.to(device)

        optimizer.zero_grad()

        output = model(input)['clipwise_output']
        # loss 默认返回该batch上所有样本损失的平均值（即reduction='mean')
        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * input.size(0)
    # 最后一个batch可能不完整
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss
def evaluate_metric(self, pred, ans):
    acc = accuracy_score(ans, np.argmax(pred, 1))
    return {"acc": acc}  
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for batch in pbar:
            input, label = batch['waveform'], batch['target']
            input, label = input.to(device), label.to(device)
            output = model(input)['clipwise_output']
            loss = criterion(output, label)
            val_loss += loss.item() * input.size(0)
            all_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    print(f"acc:{accuracy}|pre:{precision}|recall:{recall}|f1:{f1}|kappa:{kappa}")
    val_loss /= len(dataloader.dataset)
    return val_loss


def train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs, device, save_path=None):
    # 完整训练循环，包含验证和模型保存
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = validate(model, val_dataloader, criterion, device)
        print(f"Epoch{epoch+1}/{epochs}|Train Loss : {train_loss:.4f}|Val Loss : {val_loss:.4f}")

        # 保存最佳模型
        if save_path:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Best Model save to {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using DeepShip")
    # full_dataset = np.load(os.path.join(config.dataset_path, "deepship-data.npy"), allow_pickle=True)

    train_dataset = DeepShip_Dataset(
        # dataset=full_dataset,
        config=config,
        eval_mode=False
    )
    val_dataset = DeepShip_Dataset(
        # dataset=full_dataset,
        config=config,
        eval_mode=True
    )


    # 没有多卡分布式训练，sampler就可以设置为None
    train_dataloader = DataLoader(train_dataset, shuffle = False, batch_size = config.batch_size, num_workers = 8)
    val_dataloader = DataLoader(val_dataset, shuffle = False, batch_size = config.batch_size, num_workers = 8)

    model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config = config,
        depths = config.htsat_depth,
        embed_dim = config.htsat_dim, 
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )
    # ---- 加载 AudioSet 预训练权重（除分类头外） ----
    pretrain_path = "./workspace/ckpt/htsat_audioset_pretrain.ckpt"
    if os.path.exists(pretrain_path):
        print(f"Loading pretrained checkpoint from {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=device)
        pretrained_sd = checkpoint["state_dict"]
        # 去掉 sed_model. 前缀
        new_sd = {}
        for old_key, value in pretrained_sd.items():
            new_key = old_key.replace("sed_model.", "", 1)
            # 跳过分类头（AudioSet 527 类 → DeepShip 4 类）
            if new_key.startswith("head.") or new_key.startswith("tscam_conv."):
                continue
            new_sd[new_key] = value
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"Missing keys (ignored): {missing}")
        if unexpected:
            print(f"Unexpected keys (ignored): {unexpected}")
        print("Pretrained weights loaded successfully (except classifier head).")
    else:
        print(f"Pretrained checkpoint not found at {pretrain_path}, training from scratch.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=5, device=device, save_path='best_model.pth')

if __name__ == "__main__":
    main()