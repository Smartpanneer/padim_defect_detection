import os
import torch
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================

DATA_YAML = "dataset/data.yaml"   # path to your data.yaml
MODEL_NAME = "yolov8n.pt"         # change to yolov8s.pt if needed
EPOCHS = 200
IMGSZ = 640
BATCH = 16

# ==========================================
# CHECK GPU
# ==========================================

device = 0 if torch.cuda.is_available() else "cpu"

print("===================================")
print("YOLO Training Configuration")
print("===================================")
print(f"Data file: {DATA_YAML}")
print(f"Model: {MODEL_NAME}")
print(f"Epochs: {EPOCHS}")
print(f"Image size: {IMGSZ}")
print(f"Batch size: {BATCH}")
print(f"Device: {device}")
print("===================================")

# ==========================================
# CHECK FILE EXISTS
# ==========================================

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(f"data.yaml not found at: {DATA_YAML}")

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")
model = YOLO(MODEL_NAME)

# ==========================================
# START TRAINING
# ==========================================

print("Starting training...")

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=device,
    workers=4,
    optimizer="AdamW",
    pretrained=True,
    patience=100,
    verbose=True
)

print("===================================")
print("Training Complete!")
print("Best model saved at:")
print("runs/detect/train/weights/best.pt")
print("===================================")
