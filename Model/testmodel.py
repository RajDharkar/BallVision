import os

from ultralytics import YOLO
import cv2

VIDEOS_DOR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'basketball.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model.path.join = os.path.join('.', 'models', 'alpaca_detector.pt')

model = YOLO(model_path)