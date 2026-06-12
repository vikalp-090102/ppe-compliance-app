import cv2
import shutil
from ultralytics import YOLO

shutil.copy(
    "/kaggle/input/datasets/vikalp090102/yolo-v11-final-best-today-morning/best (9).pt",
    "/kaggle/working/best.pt"
)
print("Model copied!")

DETECTION_MODEL = "/kaggle/working/best.pt"
VIDEO_PATH      = "/kaggle/input/datasets/vikalp090102/test-video-7/Test_Video-7.mp4"

CLASS_NAMES = {
    0: "person", 1: "head", 2: "face", 3: "glasses",
    4: "face-mask-medical", 5: "face-guard", 6: "ear",
    7: "earmuffs", 8: "hands", 9: "gloves", 10: "foot",
    11: "shoes", 12: "safety-vest", 13: "tools",
    14: "helmet", 15: "medical-suit", 16: "safety-suit"
}

det_model = YOLO(DETECTION_MODEL)
cap       = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

print("Starting Step 3a...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    results = det_model.track(
        source  = frame,
        persist = True,
        conf    = 0.25,
        iou     = 0.5,
        tracker = "botsort.yaml",
        verbose = False
    )

    if frame_idx <= 5:
        print(f"\n--- Frame {frame_idx} ---")
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id   = int(box.cls[0])
                conf     = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"  {CLASS_NAMES[cls_id]} | track_id={track_id} | conf={conf:.2f} | box=({x1},{y1},{x2},{y2})")

    if frame_idx == 5:
        break

cap.release()
print("\nStep 3a done!")
