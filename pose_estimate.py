import cv2
from ultralytics import YOLO

POSE_MODEL = "/kaggle/input/datasets/vikalp090102/yolo-v11-pose-estimation/yolo11m-pose.pt"
VIDEO_PATH = "/kaggle/input/datasets/vikalp090102/test-video-7/Test_Video-7.mp4"

KEYPOINT_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye",
    3: "left_ear", 4: "right_ear", 5: "left_shoulder",
    6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip",
    12: "right_hip", 13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle"
}

pose_model = YOLO(POSE_MODEL)
cap        = cv2.VideoCapture(VIDEO_PATH)
frame_idx  = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    results = pose_model(frame, conf=0.3, verbose=False)

    if frame_idx <= 3:
        print(f"\n--- Frame {frame_idx} ---")
        if results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.xy.cpu().numpy()
            confs          = results[0].keypoints.conf.cpu().numpy()
            for person_idx, (kps, kp_confs) in enumerate(zip(keypoints_data, confs)):
                print(f"  Person {person_idx + 1}:")
                for kp_idx in [0, 9, 10, 15, 16]:
                    x, y = kps[kp_idx]
                    c    = kp_confs[kp_idx]
                    print(f"    {KEYPOINT_NAMES[kp_idx]}: ({int(x)}, {int(y)}) conf={c:.2f}")

    if frame_idx == 3:
        break

cap.release()
print("\nStep 3b done!")
