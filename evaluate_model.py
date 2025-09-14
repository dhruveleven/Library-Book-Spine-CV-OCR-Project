import os
from ultralytics import YOLO

def evaluate_model_on_test_set(model_path, data_yaml_path):
    """
    Loads a YOLOv8 model and evaluates its performance on the test dataset.

    Args:
        model_path (str): The local path to your trained model file (best.pt).
        data_yaml_path (str): The local path to your data.yaml file.
    """
    try:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
        if not os.path.exists(data_yaml_path):
            print(f"Error: Data YAML file not found at {data_yaml_path}")
            return

        # Load the trained YOLOv8 model
        model = YOLO(model_path)
        
        # Run evaluation on the test dataset
        # We explicitly set `split='test'` to tell the model to use the test set
        print("Starting evaluation on the test set...")
        metrics = model.val(data=data_yaml_path, split='test')

        # Print the key performance metrics
        print("\n--- Evaluation Results ---")
        print(f"Test mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"Test mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# IMPORTANT: Update these paths to your local file system
local_model_path = 'runs/detect/train/weights/best.pt'

local_data_yaml_path = 'D:/CV_Library_Project/roboflow_dataset/data.yaml'

evaluate_model_on_test_set(local_model_path, local_data_yaml_path)