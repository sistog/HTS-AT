import torch
import Deepship_config as config
from model.htsat import HTSAT_Swin_Transformer

# 1. 构建模型
model = HTSAT_Swin_Transformer(
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

# 2. 加载 DeepShip 训练好的权重
state_dict = torch.load("best_model.pth", map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"Missing keys (ignored): {missing}")
if unexpected:
    print(f"Unexpected keys (ignored): {unexpected}")
model.eval()

# 3. 包装模型：HTSAT 的 forward() 返回 dict，ONNX 需要拆成 tuple
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # 与训练验证路径一致：infer_mode=False → reshape_wav2img + forward_features
        out = self.model(x)
        return out['clipwise_output'], out['framewise_output'], out['latent_output']

wrapped_model = ModelWrapper(model)

# 4. 创建虚拟输入（用 config 中的 clip_samples 保证长度正确）
dummy_input = torch.randn(1, config.clip_samples)  # (1, 48000) for DeepShip

# 5. 导出 ONNX
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "htsat_swin_transformer.onnx",
    input_names=["input"],
    output_names=["clipwise_output", "framewise_output", "latent_output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "clipwise_output": {0: "batch_size"},
        "framewise_output": {0: "batch_size"},
        "latent_output": {0: "batch_size"},
    },
    opset_version=11,
)
print("ONNX export successful!")

# 6. 验证 ONNX 文件
import onnx
onnx_model = onnx.load("htsat_swin_transformer.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")
