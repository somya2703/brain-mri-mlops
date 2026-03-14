# Optional TensorRT optimization
# Requires supported CUDA version and TensorRT installation

try:
    import tensorrt as trt
except ImportError:
    print("TensorRT not installed. Skipping TensorRT optimization.")
    exit()

import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(TRT_LOGGER)

network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

parser = trt.OnnxParser(network, TRT_LOGGER)

with open("models/model.onnx","rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()

config.max_workspace_size = 1 << 30

engine = builder.build_engine(network,config)

with open("models/model.trt","wb") as f:
    f.write(engine.serialize())

print("TensorRT engine created")