from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms
import torch
import io
import numpy as np
import base64
import cv2

app = FastAPI(title="Defect Detection API")

checkpoint = torch.load(
    "results/Patchcore/bottle/v3/weights/torch/model.pt",
    map_location="cpu",
    weights_only=False,  # we trust this checkpoint -- we trained it ourselves
)
model = checkpoint["model"]  # already a fully fitted Patchcore model, not a state_dict
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size  # (width, height) -- to resize the heatmap back to match
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    # anomaly_map is a per-pixel score at the model's internal resolution --
    # normalize it to 0-255 and resize back to the original image size
    amap = output.anomaly_map[0].squeeze().cpu().numpy()

    # Use a fixed scale instead of per-image min/max, so a uniformly "boring"
# (non-anomalous) image doesn't get its tiny natural variation stretched
# into a dramatic-looking heatmap. These bounds come from PatchCore's
# typical anomaly score range -- values below ~0 are clipped to 0 (no
# color), values above ~1 are clipped to max color.
    FIXED_MIN, FIXED_MAX = 0.0, 1.0
    amap_clipped = np.clip(amap, FIXED_MIN, FIXED_MAX)
    amap_norm = ((amap_clipped - FIXED_MIN) / (FIXED_MAX - FIXED_MIN) * 255).astype(np.uint8)
    
    amap_resized = cv2.resize(amap_norm, original_size)

    # Apply a color map (heatmap colors) and overlay on the original image
    heatmap_color = cv2.applyColorMap(amap_resized, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

    # Encode the overlay as base64 so it can be sent back in JSON
    _, buffer = cv2.imencode(".png", overlay)
    overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "filename": file.filename,
        "pred_label": "defective" if bool(output.pred_label[0]) else "good",
        "pred_score": float(output.pred_score[0]),
        "heatmap_base64": overlay_base64,
    }


@app.get("/")
def root():
    return {"status": "Defect detection API is running"}