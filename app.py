from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import torch
import clip
from PIL import Image
import io

# Load CLIP model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    labels: List[str] = Form(...)
):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    # Tokenize labels
    text = clip.tokenize(labels).to(device)

    # Encode
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits_per_image = 100.0 * image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # Return result as dict
    return {label: float(prob) for label, prob in zip(labels, probs)}