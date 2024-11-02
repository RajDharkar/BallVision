import os
from ultralytics import YOLO

model = YOLO("C:\Users\meggs\runs\detect\train9\weights\best.pt")

image_dir = r'Ping Pong Detection.v3i.yolov\test\images' #Raw String
output_dir = r'Ping Pong Detection.v3i.yolov11\test\outputs' #r' = raw string

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)

        results = model(image_path)

        for result in results:
            masks = result.boxes
            masks = result.masks
            keypoints = result.keypoints
            probs = result.probs
            obs = result.obb

            result.show()

            output_path = os.path.join(output_dir, filename)
            result.save(filename=output_path)
