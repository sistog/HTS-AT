import numpy as np
import onnxruntime as ort
import librosa
import Deepship_config as config

# DeepShip 类别
CLASS_NAMES = ['Cargo', 'PassengerShip', 'Tanker', 'Tug']  # 注意拼写修正

# 1. 加载音频并统一长度到 clip_samples (48000)
FILE_PATH = "/data/zcx/wav_prj/PANN_Models_DeepShip-main/Datasets/DeepShip/Segments_3s_16000hz/Cargo/1/1_Cargo-Segment_3.wav"
waveform, sr = librosa.load(FILE_PATH, sr=16000)

target_len = config.clip_samples  # 48000
if len(waveform) > target_len:
    waveform = waveform[:target_len]
elif len(waveform) < target_len:
    waveform = np.pad(waveform, (0, target_len - len(waveform)))

print(f"音频长度: {len(waveform)} 采样点 ({len(waveform)/16000:.1f}s)")

# 2. ONNX Runtime 推理
ort_session = ort.InferenceSession("htsat_swin_transformer.onnx")

ort_input = waveform[None, :].astype(np.float32)  # (1, 48000)
ort_inputs = {ort_session.get_inputs()[0].name: ort_input}

# 输出顺序: [clipwise_output, framewise_output, latent_output]
ort_outputs = ort_session.run(None, ort_inputs)
logits = ort_outputs[0]  # clipwise_output, shape (1, 4)

# 3. Softmax 转概率（多分类的正确方式）
def softmax(x):
    """Compute softmax values for each row"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 数值稳定性
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

probs = softmax(logits)  # shape (1, 4)
pred_probs = probs[0]

# 4. 取 Top-3（概率最高的3个类别）
top3_indices = np.argsort(pred_probs)[-3:][::-1]

print(f"\n{'='*50}")
print(f"ONNX Runtime 推理结果 (Softmax):")
print(f"{'='*50}")
print(f"  {CLASS_NAMES[top3_indices[0]]:15s} {pred_probs[top3_indices[0]]:.4f}  ← 最高 (置信度: {pred_probs[top3_indices[0]]*100:.1f}%)")

print(f"\n  Top-3 预测:")
for i, idx in enumerate(top3_indices):
    bar = "█" * int(pred_probs[idx] * 50)
    print(f"  [{idx}] {CLASS_NAMES[idx]:15s} {pred_probs[idx]:.4f}  {bar}")

# 验证概率总和是否为 1
print(f"\n  概率总和: {np.sum(pred_probs):.6f} (应为 1.0)")

# 5. 对比 predict.py 格式输出
print(f"\n对比 predict.py 输出格式:")
predict_output = [[int(idx), CLASS_NAMES[idx], float(pred_probs[idx])] for idx in top3_indices]
print(predict_output)