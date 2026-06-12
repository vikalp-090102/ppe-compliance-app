import cv2
import numpy as np
from ultralytics import YOLO

DETECTION_MODEL = "/kaggle/working/best.pt"
POSE_MODEL      = "/kaggle/input/datasets/vikalp090102/yolo-v11-pose-estimation/yolo11m-pose.pt"
VIDEO_PATH      = "/kaggle/input/datasets/vikalp090102/test-video-7/Test_Video-7.mp4"

CLASS_NAMES = {
    0: "person", 1: "head", 2: "face", 3: "glasses",
    4: "face-mask-medical", 5: "face-guard", 6: "ear",
    7: "earmuffs", 8: "hands", 9: "gloves", 10: "foot",
    11: "shoes", 12: "safety-vest", 13: "tools",
    14: "helmet", 15: "medical-suit", 16: "safety-suit"
}

REQUIRED_PPE = {14: "helmet", 12: "safety-vest", 9: "gloves", 11: "shoes", 3: "glasses"}
PPE_CLASSES  = [3, 4, 5, 7, 9, 11, 12, 14, 15, 16]
HEAD_PPE     = [3, 7, 4, 5]
HAND_PPE     = [9, 8]
FEET_PPE     = [11, 10]

det_model  = YOLO(DETECTION_MODEL)
pose_model = YOLO(POSE_MODEL)
cap        = cv2.VideoCapture(VIDEO_PATH)
frame_idx  = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    det_results  = det_model.track(source=frame, persist=True, conf=0.25,
                                    iou=0.5, tracker="botsort.yaml", verbose=False)
    pose_results = pose_model(frame, conf=0.3, verbose=False)

    persons   = []
    ppe_items = []

    if det_results[0].boxes is not None:
        for box in det_results[0].boxes:
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cls_id == 0:
                persons.append({
                    "track_id" : track_id,
                    "box"      : (x1, y1, x2, y2),
                    "cx"       : cx,
                    "cy"       : cy,
                    "keypoints": None,
                    "nose_cy"  : None,
                    "ppe"      : []
                })
            elif cls_id in PPE_CLASSES:
                ppe_items.append({
                    "cls_id": cls_id,
                    "conf"  : conf,
                    "cx"    : cx,
                    "cy"    : cy,
                })

    if pose_results[0].keypoints is not None:
        kps_data  = pose_results[0].keypoints.xy.cpu().numpy()
        kps_confs = pose_results[0].keypoints.conf.cpu().numpy()
        for i, person in enumerate(persons):
            if i < len(kps_data):
                valid_kps = [
                    kps_data[i][kp_idx]
                    for kp_idx in [0, 9, 10, 15, 16]
                    if kps_confs[i][kp_idx] > 0.1
                ]
                persons[i]["keypoints"] = valid_kps if valid_kps else None
                if kps_confs[i][0] > 0.1:
                    persons[i]["nose_cy"] = float(kps_data[i][0][1])

    if len(persons) > 0 and len(ppe_items) > 0:
        for ppe in ppe_items:
            best_person = None
            best_dist   = 9999

            for j, person in enumerate(persons):
                if ppe["cls_id"] in HEAD_PPE:
                    if person["keypoints"]:
                        dist = np.sqrt(
                            (ppe["cx"] - person["keypoints"][0][0])**2 +
                            (ppe["cy"] - person["keypoints"][0][1])**2
                        )
                    else:
                        dist = np.sqrt(
                            (ppe["cx"] - person["cx"])**2 +
                            (ppe["cy"] - person["cy"])**2
                        )

                elif ppe["cls_id"] in HAND_PPE:
                    if person["keypoints"] and len(person["keypoints"]) >= 3:
                        wrists = person["keypoints"][1:3]
                        dists  = [
                            np.sqrt((ppe["cx"] - kp[0])**2 + (ppe["cy"] - kp[1])**2)
                            for kp in wrists if kp[0] > 0
                        ]
                        dist = min(dists) if dists else 9999
                    else:
                        dist = np.sqrt(
                            (ppe["cx"] - person["cx"])**2 +
                            (ppe["cy"] - person["cy"])**2
                        )

                elif ppe["cls_id"] in FEET_PPE:
                    if person["keypoints"] and len(person["keypoints"]) > 3:
                        ankles = person["keypoints"][3:]
                        dists  = [
                            np.sqrt((ppe["cx"] - kp[0])**2 + (ppe["cy"] - kp[1])**2)
                            for kp in ankles if kp[0] > 0
                        ]
                        dist = min(dists) if dists else 9999
                    else:
                        dist = np.sqrt(
                            (ppe["cx"] - person["cx"])**2 +
                            (ppe["cy"] - person["cy"])**2
                        )

                else:
                    dist = np.sqrt(
                        (ppe["cx"] - person["cx"])**2 +
                        (ppe["cy"] - person["cy"])**2
                    )

                if dist < best_dist:
                    best_dist   = dist
                    best_person = j

            if best_person is not None and best_dist < 400:
                persons[best_person]["ppe"].append(ppe["cls_id"])

    if frame_idx <= 5:
        print(f"\n--- Frame {frame_idx} ---")
        for person in persons:
            detected = [CLASS_NAMES[k] for k in set(person["ppe"])]
            missing  = [REQUIRED_PPE[k] for k in REQUIRED_PPE if k not in set(person["ppe"])]
            print(f"  Person track_id={person['track_id']}")
            print(f"    Detected PPE : {detected}")
            print(f"    Missing PPE  : {missing}")
            print(f"    Alert        : {'YES' if missing else 'NO'}")

    if frame_idx == 5:
        break

cap.release()
print("\nStep 3c done!")
