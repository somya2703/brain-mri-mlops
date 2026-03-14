import os
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd

MODEL_PATH = "models/model.onnx"
DATA_DIR = "data/raw"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print("Using providers:", session.get_providers())
input_name = session.get_inputs()[0].name

def preprocess(image):

    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.transpose(image,(2,0,1))
    image = np.expand_dims(image,axis=0)

    return image.astype(np.float32)

results = []

for label in os.listdir(DATA_DIR):
    import time
    time.sleep(0.3)

    folder = os.path.join(DATA_DIR,label)

    for img_name in os.listdir(folder):

        path = os.path.join(folder,img_name)

        image = Image.open(path).convert("RGB")

        input_tensor = preprocess(image)

        output = session.run(None, {input_name: input_tensor})

        pred = np.argmax(output[0])
        conf = float(np.max(output[0]))

        results.append({
            "image": img_name,
            "true_label": label,
            "prediction": int(pred),
            "confidence": conf
        })

df = pd.DataFrame(results)

df.to_csv("batch_predictions.csv", index=False)

print("Batch inference completed")
