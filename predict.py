from typing import Any
from cog import BasePredictor, Input, Path
from model.htsat import HTSAT_Swin_Transformer
import torch
import librosa
import numpy as np
import pandas as pd

import Deepship_config as config
SAMPLE_RATE = 16000


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # preprocess the class_label_indice
        self.idx_2_label = {0:'Cargo', 1:'PassengerShip', 2:'Tanker', 3:'Tug'}

        # load model
        state_dict = torch.load(
            "/data/zcx/wav_prj/HTS-Audio-Transformer/best_model.pth", 
            map_location=torch.device("cpu")
        )
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
            num_heads=config.htsat_num_head,
        )
        self.sed_model.load_state_dict(state_dict)
        self.sed_model.eval()

    # Define the arguments and types the model takes as input
    def predict(self, audio: Path = Input(description="Audio to classify")) -> Any:
        """Run a single prediction on the model"""
        # Preprocess the audio
        waveform, sr = librosa.load(audio, sr=SAMPLE_RATE)

        with torch.no_grad():
            x = torch.from_numpy(waveform).float()
            output_dict = self.sed_model(x[None, :])
            
            # 获取 logits (原始输出)
            logits = output_dict["clipwise_output"]  # shape: (1, 4)
            
            # 使用 softmax 转换为概率分布 (总和为1)
            probs = torch.softmax(logits, dim=1)  # shape: (1, 4)
            probs = probs[0].detach().cpu().numpy()  # shape: (4,)
            
            # 获取 top-3 预测
            pred_labels = np.argsort(probs)[-3:][::-1]  # 概率最高的3个类别
            
        return [
            [int(pred_label), self.idx_2_label[pred_label], float(probs[pred_label])]
            for pred_label in pred_labels
        ]


if __name__ == "__main__":
    p = Predictor()
    p.setup()  # 加载模型

    result = p.predict(Path("/data/zcx/wav_prj/PANN_Models_DeepShip-main/Datasets/DeepShip/Segments_3s_16000hz/Cargo/6/6_Cargo-Segment_4.wav"))
    print(result)