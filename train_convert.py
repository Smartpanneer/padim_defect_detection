import os
import random
import shutil
from collections import defaultdict

# === CONFIG ===
images_dir = 'dataset/images'
labels_dir = 'dataset/labels'
output_dir = 'output_yolo_split'
train_ratio = 0.8  # 80% train, 20% val

# === CREATE OUTPUT DIRS ===
train_img_dir = os.path.join(output_dir, 'train/images')
train_lbl_dir = os.path.join(output_dir, 'train/labels')
val_img_dir   = os.path.join(output_dir, 'valid/images')
val_lbl_dir   = os.path.join(output_dir, 'valid/labels')

for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# === GROUP IMAGES BY CLASS ===
class_to_files = defaultdict(list)

for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    base = os.path.splitext(img_file)[0]
    lbl_file = f"{base}.txt"
    lbl_path = os.path.join(labels_dir, lbl_file)

    if not os.path.exists(lbl_path):
        print(f"⚠️ Warning: no label for {img_file}")
        continue

    with open(lbl_path, 'r') as f:
        lines = f.readlines()

    classes_in_img = set(int(line.split()[0]) for line in lines if line.strip())

    for cls in classes_in_img:
        class_to_files[cls].append((img_file, lbl_file))

# === SPLIT PER CLASS ===
train_files, val_files = set(), set()

for cls, files in class_to_files.items():
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    train_subset = files[:split_idx]
    val_subset = files[split_idx:]

    train_files.update(train_subset)
    val_files.update(val_subset)

def copy_pairs(pairs, img_dest, lbl_dest):
    for img_file, lbl_file in pairs:
        shutil.copy2(os.path.join(images_dir, img_file), os.path.join(img_dest, img_file))
        shutil.copy2(os.path.join(labels_dir, lbl_file), os.path.join(lbl_dest, lbl_file))

copy_pairs(train_files, train_img_dir, train_lbl_dir)
copy_pairs(val_files, val_img_dir, val_lbl_dir)

print("✅ Stratified dataset split complete.")
