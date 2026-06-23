"""
Tutorial on training a HTS-AT model for audio classification on the ESC-50 Dataset

Reference:
HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection, ICASSP 2022
https://arxiv.org/abs/2202.00874
"""

# ============================================================
# Cell 1: Imports and environment setup
# ============================================================
import os
import numpy as np
import sys
import zipfile
import librosa
import soundfile as sf

# in the training script, we only can use one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ============================================================
# Cell 2: Build workspace and download needed files
# ============================================================
def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


workspace = "./workspace"
dataset_path = os.path.join(workspace, "esc-50")
checkpoint_path = os.path.join(workspace, "ckpt")
esc_raw_path = os.path.join(dataset_path, 'raw')

create_path(workspace)
create_path(dataset_path)
create_path(checkpoint_path)
create_path(esc_raw_path)


# ============================================================
# Cell 4: Load model packages
# ============================================================
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings

from utils import create_folder, dump_config, process_idc
import esc_config as config
from sed_model import SEDWrapper, Ensemble_SEDWrapper
from data_generator import ESC_Dataset, DeepShip_Dataset
from model.htsat import HTSAT_Swin_Transformer


# ============================================================
# Cell 5: Data Preparation
# ============================================================
class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, device_num):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device_num = device_num

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle=False) if self.device_num > 1 else None
        train_loader = DataLoader(
            dataset=self.train_dataset,
            num_workers=config.num_workers,
            batch_size=config.batch_size // self.device_num,
            shuffle=False,
            sampler=train_sampler
        )
        return train_loader

    def val_dataloader(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False) if self.device_num > 1 else None
        eval_loader = DataLoader(
            dataset=self.eval_dataset,
            num_workers=config.num_workers,
            batch_size=config.batch_size // self.device_num,
            shuffle=False,
            sampler=eval_sampler
        )
        return eval_loader

    def test_dataloader(self):
        test_sampler = DistributedSampler(self.eval_dataset, shuffle=False) if self.device_num > 1 else None
        test_loader = DataLoader(
            dataset=self.eval_dataset,
            num_workers=config.num_workers,
            batch_size=config.batch_size // self.device_num,
            shuffle=False,
            sampler=test_sampler
        )
        return test_loader


# ============================================================
# Cell 6: Set the workspace and dataset
# ============================================================
device_num = torch.cuda.device_count()
print("each batch size:", config.batch_size // device_num)

exp_dir = os.path.join(config.workspace, "results", config.exp_name)
checkpoint_dir = os.path.join(config.workspace, "results", config.exp_name, "checkpoint")
if not config.debug:
    create_folder(os.path.join(config.workspace, "results"))
    create_folder(exp_dir)
    create_folder(checkpoint_dir)
    dump_config(config, os.path.join(exp_dir, config.exp_name), False)

print("Using ESC-50")
# Load the full ESC-50 dataset (5 folds)
full_dataset = np.load(os.path.join(config.dataset_path, "esc-50-data.npy"), allow_pickle=True)

dataset = ESC_Dataset(
    dataset=full_dataset,
    config=config,
    eval_mode=False
)
eval_dataset = ESC_Dataset(
    dataset=full_dataset,
    config=config,
    eval_mode=True
)

audioset_data = data_prep(dataset, eval_dataset, device_num)
checkpoint_callback = ModelCheckpoint(
    monitor="acc",
    filename='l-{epoch:d}-{acc:.3f}',
    save_top_k=20,
    mode="max"
)


# ============================================================
# Cell 7: Set the Trainer and model
# ============================================================
trainer = pl.Trainer(
    deterministic=False,
    default_root_dir=checkpoint_dir,
    gpus=device_num,
    val_check_interval=1.0,
    max_epochs=config.max_epoch,
    auto_lr_find=True,
    sync_batchnorm=True,
    callbacks=[checkpoint_callback],
    accelerator="ddp" if device_num > 1 else None,
    num_sanity_val_steps=0,
    resume_from_checkpoint=None,
    replace_sampler_ddp=False,
    gradient_clip_val=1.0
)

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

model = SEDWrapper(
    sed_model=sed_model,
    config=config,
    dataset=dataset
)

if config.resume_checkpoint is not None:
    print("Load Checkpoint from ", config.resume_checkpoint)
    ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
    ckpt["state_dict"].pop("sed_model.head.weight")
    ckpt["state_dict"].pop("sed_model.head.bias")
    # finetune on the esc and spv2 dataset
    ckpt["state_dict"].pop("sed_model.tscam_conv.weight")
    ckpt["state_dict"].pop("sed_model.tscam_conv.bias")
    model.load_state_dict(ckpt["state_dict"], strict=False)


# ============================================================
# Cell 8: Training the model
# ============================================================
if __name__ == '__main__':
    trainer.fit(model, audioset_data)


# ============================================================
# Cell 10: Inference - Audio Classification
# ============================================================
class Audio_Classification:
    def __init__(self, model_path, config):
        super().__init__()

        self.device = torch.device('cuda')
        self.sed_model = HTSAT_Swin_Transformer(
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
        ckpt = torch.load(model_path, map_location="cpu")
        temp_ckpt = {}
        for key in ckpt["state_dict"]:
            temp_ckpt[key[10:]] = ckpt['state_dict'][key]
        self.sed_model.load_state_dict(temp_ckpt)
        self.sed_model.to(self.device)
        self.sed_model.eval()

    def predict(self, audiofile):
        if audiofile:
            waveform, sr = librosa.load(audiofile, sr=32000)

            with torch.no_grad():
                x = torch.from_numpy(waveform).float().to(self.device)
                output_dict = self.sed_model(x[None, :], None, True)
                pred = output_dict['clipwise_output']
                pred_post = pred[0].detach().cpu().numpy()
                pred_label = np.argmax(pred_post)
                pred_prob = np.max(pred_post)
            return pred_label, pred_prob


# ============================================================
# Cell 11: Inference execution (example usage)
# ============================================================
# Uncomment and fill in the paths to run inference:
#
# model_path = 'path/to/your/saved/model.ckpt'
# meta_path = 'path/to/your/meta.csv'
#
# # get the groundtruth
# meta = np.loadtxt(meta_path, delimiter=',', dtype='str', skiprows=1)
# gd = {}
# for label in meta:
#     name = label[0]
#     target = label[2]
#     gd[name] = target
#
# Audiocls = Audio_Classification(model_path, config)
#
# # pick any audio you like in the testing set
# pred_label, pred_prob = Audiocls.predict("./path/to/audio.wav")
#
# print('Audiocls predict output: ', pred_label, pred_prob, gd["audio_filename.wav"])
