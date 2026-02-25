import cv2
from ultralytics import YOLO
import os

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = "runs/detect/train/weights/best.pt"
IMAGE_PATH = "test_4.jpg"
CONF_THRESHOLD = 0.8
OUTPUT_PATH = "yolo_output.jpg"

# ==========================================
# CHECK FILES
# ==========================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found!")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError("Image not found!")

# ==========================================
# LOAD MODEL
# ==========================================
model = YOLO(MODEL_PATH)

# ==========================================
# RUN DETECTION
# ==========================================
results = model.predict(
    source=IMAGE_PATH,
    conf=CONF_THRESHOLD,
    save=False
)

# ==========================================
# USE YOLO BUILT-IN PLOT (CORRECT BOXES)
# ==========================================
annotated_image = results[0].plot()  # <-- correct scaled image

# Save output
cv2.imwrite(OUTPUT_PATH, annotated_image)

print("Saved:", OUTPUT_PATH)