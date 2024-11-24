import torch
from ultralytics import YOLO
# Load the YOLOv8 model
model = YOLO("yolov8n.yaml")

# Path to the dataset
data_path = "dataset.yaml"  # Replace this with the actual path to your dataset.yaml file

# Train the model
results = model.train(
    data=data_path,  # Dataset configuration
    epochs=1,       # Number of training epochs
    batch=4,        # Batch size
    imgsz=160,       # Image size
    name="drone_detection"  # Folder name for saving results
)

# Save the best model manually by using torch.save
best_model_path = r"C:\Users\ADMIN\runs\detect\drone_detection3\weights\best.pt"  # Path where YOLOv8 saves the best model
model_weights = torch.load(best_model_path)  # Load the model weights
torch.save(model_weights, "drone_yolov8_model.pt")  # Save the model manually to a new file

print("Model saved successfully!")

# Validate the model
metrics = model.val()
print(f"Validation Results: {metrics}")

# Test detection on sample images
test_results = model.predict(source=r"C:\Users\ADMIN\Desktop\DRONE\dataset_txt\images\test", save=True)  # Replace 'path_to_test_images' with your folder
