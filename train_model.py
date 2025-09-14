from ultralytics import YOLO

# Load a pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model on your custom dataset
# The 'data.yaml' file specifies where your data is located.
# We set 'epochs' to 100 to give the model plenty of time to learn,
# and 'imgsz' to 640, a standard size for object detection.
results = model.train(data='data.yaml', epochs=100, imgsz=640)