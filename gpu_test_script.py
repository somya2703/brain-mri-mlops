import onnxruntime as ort
import numpy as np
import time

MODEL_PATH = "models/model.onnx"

cuda_options = {
    "device_id": 0,
    "arena_extend_strategy": "kNextPowerOfTwo",
    "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
    "cudnn_conv_algo_search": "EXHAUSTIVE",
}

session = ort.InferenceSession(
    MODEL_PATH,
    providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
)

print("Using providers:", session.get_providers())

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print("Model expected input shape:", input_shape)

input_shape = [1 if x is None else x for x in input_shape]

dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

print("Starting GPU stress inference...")

while True:
    session.run(None, {input_name: dummy_input})
    time.sleep(0.05)