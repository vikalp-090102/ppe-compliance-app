import os
import shutil
import random
from collections import defaultdict

random.seed(42)

# ── PATHS ──────────────────────────────────────────────────────────────────
SH17_BASE    = "/kaggle/input/datasets/mugheesahmad/sh17-dataset-for-ppe-detection"
HARDHAT_BASE = "/kaggle/input/datasets/vikalp090102/roboflow-helmet"
GLOVE_BASE   = "/kaggle/input/datasets/vikalp090102/roboflow-glove"
ULTRA_BASE   = "/kaggle/input/datasets/vikalp090102/ultralytics-constuction"
OUTPUT_BASE  = "/kaggle/working/merged_dataset"

HARDHAT_MAP = {0: 1, 1: 14, 2: 0}
GLOVE_MAP   = {0: 9}
ULTRA_MAP   = {0: 14, 1: 9, 2: 12, 3: 11, 4: 3, 6: 0}
ULTRA_SKIP  = {5, 7, 8, 9, 10}

os.makedirs(f"{OUTPUT_BASE}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/labels", exist_ok=True)

train_files = []
val_files   = []

# ── HELPERS ────────────────────────────────────────────────────────────────
def remap_label(src_lbl, dst_lbl, class_map, skip=None):
    with open(src_lbl) as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            old_cls = int(parts[0])
            if skip and old_cls in skip:
                continue
            if old_cls in class_map:
                parts[0] = str(class_map[old_cls])
                new_lines.append(" ".join(parts))
    if new_lines:
        with open(dst_lbl, "w") as f:
            f.write("\n".join(new_lines))
        return True
    return False

def copy_sh17_label_no_helmet(src_lbl, dst_lbl):
    """Copy SH17 label but remove helmet (class 14) annotations"""
    with open(src_lbl) as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts and int(parts[0]) != 14:
            new_lines.append(line.strip())
    if new_lines:
        with open(dst_lbl, "w") as f:
            f.write("\n".join(new_lines))
        return True
    return False

def get_classes(lbl_path):
    classes = set()
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    return classes

def stratified_sample(img_dir, lbl_dir, class_map, target_count):
    all_imgs = [f for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    groups = defaultdict(list)
    for img_file in all_imgs:
        base     = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(lbl_dir, f"{base}.txt")
        if not os.path.exists(lbl_path):
            continue
        classes   = get_classes(lbl_path)
        if class_map:
            classes = {class_map.get(c, c) for c in classes}
        group_key = tuple(sorted(classes))
        groups[group_key].append(img_file)

    print(f"  Found {len(groups)} unique class combinations")
    print(f"  Total images: {len(all_imgs)}")

    ratio    = min(1.0, target_count / len(all_imgs))
    selected = []
    for group_key, imgs in groups.items():
        n = max(1, int(len(imgs) * ratio))
        selected.extend(random.sample(imgs, min(n, len(imgs))))

    print(f"  Selected: {len(selected)} images (ratio={ratio:.2f})")
    return selected

def copy_selected(img_dir, lbl_dir, class_map, selected_imgs, split, skip=None):
    count = 0
    for img_file in selected_imgs:
        base    = os.path.splitext(img_file)[0]
        src_img = os.path.join(img_dir, img_file)
        src_lbl = os.path.join(lbl_dir, f"{base}.txt")

        if not os.path.exists(src_lbl):
            continue

        dst_img = os.path.join(OUTPUT_BASE, "images", img_file)
        dst_lbl = os.path.join(OUTPUT_BASE, "labels", f"{base}.txt")

        shutil.copy(src_img, dst_img)

        if class_map:
            success = remap_label(src_lbl, dst_lbl, class_map, skip)
        else:
            shutil.copy(src_lbl, dst_lbl)
            success = True

        if success:
            if split == "train":
                train_files.append(dst_img)
            else:
                val_files.append(dst_img)
            count += 1

    return count

# ── STEP 1: SH17 (helmet annotations removed) ─────────────────────────────
print("Processing SH17 (removing helmet annotations)...")
with open(f"{SH17_BASE}/train_files.txt") as f:
    sh17_train = [l.strip() for l in f if l.strip()]
with open(f"{SH17_BASE}/val_files.txt") as f:
    sh17_val   = [l.strip() for l in f if l.strip()]

for img_name in sh17_train:
    base    = os.path.splitext(img_name)[0]
    src_img = os.path.join(SH17_BASE, "images", img_name)
    src_lbl = os.path.join(SH17_BASE, "labels", f"{base}.txt")
    dst_img = os.path.join(OUTPUT_BASE, "images", img_name)
    dst_lbl = os.path.join(OUTPUT_BASE, "labels", f"{base}.txt")
    if os.path.exists(src_img) and os.path.exists(src_lbl):
        shutil.copy(src_img, dst_img)
        success = copy_sh17_label_no_helmet(src_lbl, dst_lbl)
        if success:
            train_files.append(dst_img)

for img_name in sh17_val:
    base    = os.path.splitext(img_name)[0]
    src_img = os.path.join(SH17_BASE, "images", img_name)
    src_lbl = os.path.join(SH17_BASE, "labels", f"{base}.txt")
    dst_img = os.path.join(OUTPUT_BASE, "images", img_name)
    dst_lbl = os.path.join(OUTPUT_BASE, "labels", f"{base}.txt")
    if os.path.exists(src_img) and os.path.exists(src_lbl):
        shutil.copy(src_img, dst_img)
        success = copy_sh17_label_no_helmet(src_lbl, dst_lbl)
        if success:
            val_files.append(dst_img)

print(f"  SH17 train: {len(train_files)} | val: {len(val_files)}")

# ── STEP 2: HARDHAT stratified ────────────────────────────────────────────
print("\nProcessing Hardhat (stratified)...")
h_selected_train = stratified_sample(
    f"{HARDHAT_BASE}/train/images",
    f"{HARDHAT_BASE}/train/labels",
    HARDHAT_MAP, target_count=1500
)
h_selected_val = stratified_sample(
    f"{HARDHAT_BASE}/valid/images",
    f"{HARDHAT_BASE}/valid/labels",
    HARDHAT_MAP, target_count=375
)
h_train = copy_selected(f"{HARDHAT_BASE}/train/images", f"{HARDHAT_BASE}/train/labels", HARDHAT_MAP, h_selected_train, "train")
h_val   = copy_selected(f"{HARDHAT_BASE}/valid/images", f"{HARDHAT_BASE}/valid/labels", HARDHAT_MAP, h_selected_val,   "val")
print(f"  Hardhat train: {h_train} | val: {h_val}")

# ── STEP 3: GLOVE (100%) ──────────────────────────────────────────────────
print("\nProcessing Glove (100%)...")
g_selected_train = stratified_sample(
    f"{GLOVE_BASE}/train/images",
    f"{GLOVE_BASE}/train/labels",
    GLOVE_MAP, target_count=3378
)
g_selected_val = stratified_sample(
    f"{GLOVE_BASE}/valid/images",
    f"{GLOVE_BASE}/valid/labels",
    GLOVE_MAP, target_count=122
)
g_selected_test = stratified_sample(
    f"{GLOVE_BASE}/test/images",
    f"{GLOVE_BASE}/test/labels",
    GLOVE_MAP, target_count=42
)
g_train = copy_selected(f"{GLOVE_BASE}/train/images", f"{GLOVE_BASE}/train/labels", GLOVE_MAP, g_selected_train, "train")
g_val   = copy_selected(f"{GLOVE_BASE}/valid/images", f"{GLOVE_BASE}/valid/labels", GLOVE_MAP, g_selected_val,   "val")
g_test  = copy_selected(f"{GLOVE_BASE}/test/images",  f"{GLOVE_BASE}/test/labels",  GLOVE_MAP, g_selected_test,  "val")
print(f"  Glove train: {g_train} | val: {g_val + g_test}")

# ── STEP 4: ULTRALYTICS (100%) ────────────────────────────────────────────
print("\nProcessing Ultralytics Construction-PPE (100%)...")
u_selected_train = stratified_sample(
    f"{ULTRA_BASE}/images/train",
    f"{ULTRA_BASE}/labels/train",
    ULTRA_MAP, target_count=1132
)
u_selected_val = stratified_sample(
    f"{ULTRA_BASE}/images/val",
    f"{ULTRA_BASE}/labels/val",
    ULTRA_MAP, target_count=143
)
u_selected_test = stratified_sample(
    f"{ULTRA_BASE}/images/test",
    f"{ULTRA_BASE}/labels/test",
    ULTRA_MAP, target_count=141
)
u_train = copy_selected(f"{ULTRA_BASE}/images/train", f"{ULTRA_BASE}/labels/train", ULTRA_MAP, u_selected_train, "train", skip=ULTRA_SKIP)
u_val   = copy_selected(f"{ULTRA_BASE}/images/val",   f"{ULTRA_BASE}/labels/val",   ULTRA_MAP, u_selected_val,   "val",   skip=ULTRA_SKIP)
u_test  = copy_selected(f"{ULTRA_BASE}/images/test",  f"{ULTRA_BASE}/labels/test",  ULTRA_MAP, u_selected_test,  "val",   skip=ULTRA_SKIP)
print(f"  Ultralytics train: {u_train} | val: {u_val + u_test}")

# ── STEP 5: WRITE TXT FILES ───────────────────────────────────────────────
print("\nWriting txt files...")
with open(f"{OUTPUT_BASE}/train.txt", "w") as f:
    f.write("\n".join(train_files))
with open(f"{OUTPUT_BASE}/val.txt", "w") as f:
    f.write("\n".join(val_files))

print(f"\n✅ Total train: {len(train_files)}")
print(f"✅ Total val:   {len(val_files)}")
print(f"✅ Saved in:    {OUTPUT_BASE}")
