import torch
import numpy as np

# Load mô hình JIT
model = torch.jit.load("bestbest.pt", map_location=torch.device("cpu"))
model.eval()

# Dummy input (phải có đúng kích thước với mô hình)
dummy_input = torch.randn(1, 3, 260, 260, dtype=torch.float32)

# Xuất sang ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,  # Bạn có thể dùng 11, 13 hoặc cao hơn tùy vào onnxruntime
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("Mô hình ONNX hợp lệ!")
