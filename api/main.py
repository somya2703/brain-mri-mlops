import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from PIL import Image
import io
import time

from prometheus_client import Counter, Histogram
from prometheus_client import generate_latest

app = FastAPI()

# Prometheus metrics

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

LATENCY = Histogram(
    "prediction_latency_seconds",
    "Inference latency in seconds"
)

session = ort.InferenceSession(
    "models/model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

def preprocess(image):

    image = image.resize((224,224))

    image = np.array(image).astype(np.float32) / 255.0

    # HWC -> CHW
    image = np.transpose(image, (2,0,1))

    # add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

@app.post("/predict")

async def predict(file: UploadFile):

    start = time.time()

    REQUEST_COUNT.inc()

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = preprocess(image)

    outputs = session.run(None,{"input":input_tensor})

    logits = outputs[0]

    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    probs = probs[0]

    pred = int(np.argmax(probs))
    confidence = float(np.max(probs))
    '''
    pred = int(np.argmax(outputs[0]))

    confidence = float(np.max(outputs[0]))
    '''
    LATENCY.observe(time.time() - start)

    #label = "tumor" if pred==1 else "no_tumor"
    label_map = {
        0: "no_tumor",
        1: "tumor"
    }

    label = label_map[pred]

    return {
        "prediction":label,
        "confidence":confidence,
        "probabilities": {
            "no_tumor": float(probs[0]),
            "tumor": float(probs[1])
        }  
    }
@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )
