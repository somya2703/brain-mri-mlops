import onnxruntime as ort
import numpy as np
from PIL import Image

# load model
session = ort.InferenceSession("models/model.onnx")

def preprocess(image):

    image = image.resize((224,224))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image,(2,0,1))
    image = np.expand_dims(image,axis=0)

    return image

# test tumor image
image_path = "data/raw/yes/Y1.jpg"

image = Image.open(image_path).convert("RGB")

input_tensor = preprocess(image)

outputs = session.run(None, {"input": input_tensor})

logits = outputs[0]

exp_logits = np.exp(logits)
probs = exp_logits / np.sum(exp_logits)

print("\nRaw logits:", logits)
print("Probabilities:", probs)
print("Predicted class:", np.argmax(probs))