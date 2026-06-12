from ultralytics import YOLO

model = YOLO("/kaggle/input/datasets/vikalp090102/yolo-11m/yolo11m.pt")

model.train(
    data         = "/kaggle/working/merged_dataset/merged.yaml",
    epochs       = 70,
    imgsz        = 640,
    batch        = -1,
    optimizer    = "AdamW",
    lr0          = 0.001,
    cos_lr       = True,
    cls          = 2.0,
    mosaic       = 1.0,
    mixup        = 0.1,
    degrees      = 10.0,
    fliplr       = 0.5,
    hsv_h        = 0.015,
    hsv_s        = 0.7,
    hsv_v        = 0.4,
    warmup_epochs = 3,
    close_mosaic  = 10,
    project      = "/kaggle/working/ppe_runs",
    name         = "yolo11m_final",
    exist_ok     = True,
)
