from ultralytics import YOLO
import random
import os

# Load a pretrained YOLO model
model = YOLO("yolo11n.yaml")  # Change "yolo11n.yaml" to a pretrained YOLO model if available

# Create a subset of your data
data_path = r"C:\Users\meggs\OneDrive\Documents\GitHub\DroneVision\Ping Pong Detection.v3i.yolov11\data.yaml"
subset_ratio = 0.3  # Use 30% of the data

# Train the model with reduced data and epochs
train_results = model.train(
    data=data_path,
    epochs=30,  # Reduced epochs
    imgsz=320,  # Reduced image size for faster training
    save_period=5  # Save checkpoints every 5 epochs
)
