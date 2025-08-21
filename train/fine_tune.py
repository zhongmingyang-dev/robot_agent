import pickle
from ultralytics import YOLO

# Load a model
model = YOLO("../models/yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="my.yaml", epochs=100, imgsz=640)

with open("test.pkl", "wb") as pkl:
    pickle.dump(results, pkl)
