import torch
from model import BrainCNN

model = BrainCNN()

model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))

model.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",

    export_params=True,
    opset_version=18,

    do_constant_folding=True,

    input_names=["input"],
    output_names=["output"],

    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("ONNX model exported with dynamic batching")