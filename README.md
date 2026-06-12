# PPE Compliance Detection System

A construction site safety monitoring system that detects whether workers are wearing required Personal Protective Equipment (PPE) using computer vision. The system processes video footage, tracks individual workers, identifies missing PPE, generates an annotated output video, and logs compliance records to a database.

---

## Target PPE Classes

The system monitors five required PPE items for each detected worker:

- Helmet
- Safety Vest
- Gloves
- Shoes
- Glasses

---

## Tech Stack Overview

### 1. Dataset Preparation

Four datasets were merged to form the final training set of 12,481 training images and 2,441 validation images:

| Dataset | Train Images | Val Images | Key Classes | Notes |
|---|---|---|---|---|
| SH17 | 6,477 | 1,620 | All 17 PPE classes | Helmet annotations removed (93% were in bottom half of images) |
| Roboflow Hardhat | 1,498 | 373 | Helmet, Head, Person | 39% stratified sample |
| Roboflow Glove | 3,378 | 164 | Gloves | 100% included |
| Ultralytics PPE | 1,128 | 284 | Helmet, Gloves, Vest, Shoes, Glasses | 100% included |

A key data cleaning step was removing all helmet annotations from the SH17 dataset. Investigation revealed that 93.3% of SH17 helmet labels were located in the bottom half of images, meaning the model was learning that helmets appear near the ground. Removing these labels and relying on Hardhat and Ultralytics datasets for helmet supervision corrected this bias.

For the Roboflow Hardhat dataset, stratified sampling was applied to retain only 39% of images. Images were grouped by their class combinations and sampled proportionally from each group, ensuring all helmet styles and camera angles were represented without introducing too much helmet-heavy data that could unbalance the merged dataset.

---

### 2. Feature Engineering

Two categories of feature engineering were applied during training: data augmentation and dataset-level engineering.

**Data Augmentation**

The following augmentation parameters were set during training to improve generalization across varied construction site conditions:

| Parameter | Value | Purpose |
|---|---|---|
| mosaic | 1.0 | Combines four images into one tile, exposing the model to multiple scenes and scales simultaneously |
| mixup | 0.1 | Blends two images together at a low rate to improve robustness to overlapping objects |
| degrees | 10.0 | Randomly rotates images up to 10 degrees to handle slightly tilted camera angles |
| fliplr | 0.5 | Horizontally flips images 50% of the time to remove left/right bias |
| hsv_h | 0.015 | Applies small hue shifts to handle lighting color variation across sites |
| hsv_s | 0.7 | Varies saturation strongly to simulate different lighting intensities |
| hsv_v | 0.4 | Varies brightness to handle shadows, overexposure, and indoor vs outdoor conditions |

Mosaic augmentation at 1.0 was particularly important because it forces the model to detect small PPE items like gloves and glasses within crowded, multi-person scenes, which closely mirrors real construction site footage.

**Dataset-Level Engineering**

The Roboflow Hardhat dataset was included using stratified sampling at 39% rather than the full set. Stratified sampling groups images by their class label combinations and samples proportionally from each group, preserving diversity in helmet appearance, angle, and occlusion level while preventing the merged dataset from becoming helmet-heavy relative to other PPE classes.

---

### 3. Model Training

**Model:** YOLOv11m (medium variant, 20 million parameters)  
**Framework:** Ultralytics  
**Epochs:** 70  
**Confidence threshold:** 0.25  
**IOU threshold:** 0.5  

YOLOv11m was chosen over the smaller YOLOv11s (9.4 million parameters) because the larger capacity allows the model to learn more complex visual patterns, particularly for small and partially visible items like gloves and glasses.

**Final model performance on key PPE classes:**

| Class | mAP50 (Baseline YOLOv11s) | mAP50 (Final YOLOv11m) |
|---|---|---|
| Helmet | 0.590 | 0.925 |
| Gloves | 0.561 | 0.757 |
| Safety Vest | 0.900 | 0.905 |
| Shoes | 0.844 | 0.865 |
| Glasses | 0.904 | 0.913 |
| Overall mAP50 | 0.588 | 0.650 |

---

### 3. Inference Pipeline

The inference pipeline consists of six steps run on every video frame:

**Step 1 - Person Detection and Tracking (`ppe_track.py`)**  
YOLOv11m detects all persons and PPE items in the frame. The BotSORT multi-object tracker assigns persistent track IDs to each person so they can be followed consistently across frames.

**Step 2 - Pose Estimation (`pose_estimate.py`)**  
YOLOv11m-pose detects 17 body keypoints per person including nose, wrists, and ankles. These keypoints are used in the next step to spatially associate PPE items with the correct person.

**Step 3 - PPE Association (`ppe_association.py`)**  
Each detected PPE item is assigned to the nearest person using keypoint-based distance matching. Head PPE (helmet, glasses, earmuffs, face mask) is matched to the nose keypoint. Hand PPE (gloves) is matched to the nearest wrist keypoint. Feet PPE (shoes) is matched to the nearest ankle keypoint. A maximum distance threshold of 400 pixels is applied to avoid false associations.

**Step 4 - Compliance Checking**  
After association, the system checks which of the five required PPE items are missing for each tracked person and raises an alert if any are absent.

**Step 5 - Annotated Video Generation (`ppe_visual_output.py`)**  
Bounding boxes are drawn around each person in green if fully compliant or red if any PPE is missing. The label shows the track ID and alert status. Detected and missing PPE items are listed inside the bounding box. The annotated video is saved as an MP4 file.

**Step 6 - Compliance Log (`ppe_compliance_log.py`)**  
Every frame's results are written to a CSV file with the following columns: Frame No., Time (s), Person ID, Track ID, Detected PPE, Missing PPE, and Alert status. This log is also pushed to a PostgreSQL database.

---

### 4. Deployment Stack

| Component | Technology |
|---|---|
| Model weights hosting | Hugging Face Hub (`vikalp090/ppe-compliance-yolo11m`) |
| Web application | Streamlit Cloud |
| Database | Supabase (PostgreSQL) |
| Code repository | GitHub |

**Application URL:** https://ppe-compliance-app-8qxajsrqw8nupvfd7pepdx.streamlit.app

The Streamlit application has three sections:

- **Run Detection:** Upload a construction site video, run the full inference pipeline, and download the annotated output video and compliance log CSV.
- **Compliance Dashboard:** Charts showing alert distribution and missing PPE frequency across the processed video.
- **Log History:** Full table of past detection records with filters by person ID and alert status. Records can be pushed to the Supabase database with one click.

---

### 5. Libraries and Dependencies

```
ultralytics
opencv-python
numpy
pandas
streamlit
supabase
huggingface_hub
```

---

### 6. Model Weights

The trained model weights are not included in this repository due to file size. They are hosted on Hugging Face:

```
https://huggingface.co/vikalp090/ppe-compliance-yolo11m
```

The Streamlit application downloads the weights automatically at startup.

---

### 7. Known Limitations

The model performs well on helmet detection when the camera is positioned at a top-down or overhead angle. This is because the majority of helmet training images were captured from above. At eye level or from the side, helmet detection accuracy drops. Addressing this would require collecting additional training data from eye-level construction cameras.

Processing runs on CPU on the free hosting tier. For a 25-second video at 30 FPS, total processing time is approximately 393 seconds.

---
