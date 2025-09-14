import os
import cv2
from ultralytics import YOLO
import sys

def detect_book_spines(image_path, model_path, confidence_threshold=0.25):
    """
    Loads a YOLOv8 model, detects book spines on an unseen image,
    and draws the bounding boxes on the image.

    Args:
        image_path (str): The path to the unseen bookshelf image.
        model_path (str): The path to your trained YOLOv8 model (best.pt).
        confidence_threshold (float): The minimum confidence score for a detection to be shown.
    """
    try:
        # Load the trained YOLOv8 model
        print("Loading the trained YOLOv8 model...")
        model = YOLO(model_path)
        
        # Load the original image using OpenCV
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        # Perform object detection on the image
        # The 'conf' argument sets the confidence threshold.
        print(f"Running object detection with confidence threshold: {confidence_threshold}")
        results = model.predict(source=original_image, conf=confidence_threshold)

        # Process the results and draw bounding boxes on the image
        for result in results:
            if result.boxes:
                # The 'xyxy' format gives us the top-left and bottom-right coordinates
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw a green rectangle on the original image
                    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the image with the bounding boxes
        print("Displaying the image with detected books...")
        
        # Resize image for better viewing on screen
        h, w, _ = original_image.shape
        scale_factor = 800 / max(h, w)
        resized_image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor)
        
        cv2.imshow('Detected Book Spines', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

# --- How to use this script ---
if __name__ == "__main__":
    # Check if the user provided an image path as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python detect_book_spine.py <path_to_unseen_image>")
    else:
        # The first argument is the script name, the second is the image path
        unseen_image_path = sys.argv[1]
        
        # The path to your trained model file
        trained_model_path = 'runs/detect/train/weights/best.pt'

        
        detect_book_spines(unseen_image_path, trained_model_path)