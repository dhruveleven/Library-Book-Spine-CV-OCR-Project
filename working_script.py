import cv2
import numpy as np
import pytesseract
import sys
from ultralytics import YOLO
from thefuzz import fuzz

# --- Prerequisites ---
# pip install opencv-python ultralytics pytesseract thefuzz
# Install Tesseract OCR and add it to PATH
# On Windows, set path manually:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def detect_and_read_books(image_path, model_path):
    """
    Detects book spines, crops/rotates them, and extracts text using Tesseract OCR.
    Returns a list of dicts with book titles and bounding boxes.
    """
    try:
        # Load YOLOv8 model
        model = YOLO(model_path)

        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Could not read image at {image_path}")
            return []

        # Run detection
        results = model.predict(source=original_image, save=False, verbose=False)

        detected_books = []

        for result in results:
            if result.boxes:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)

                    cropped_spine = original_image[y1:y2, x1:x2]

                    # Try both rotations (book spines are usually vertical)
                    spine_90 = cv2.rotate(cropped_spine, cv2.ROTATE_90_CLOCKWISE)
                    spine_270 = cv2.rotate(cropped_spine, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    text_90 = pytesseract.image_to_string(spine_90, lang="eng", config="--psm 6")
                    text_270 = pytesseract.image_to_string(spine_270, lang="eng", config="--psm 6")

                    # Choose whichever OCR gives more characters
                    extracted_text = text_90 if len(text_90) > len(text_270) else text_270

                    if extracted_text.strip():
                        detected_books.append({
                            "bbox": (x1, y1, x2, y2),
                            "text": extracted_text.strip()
                        })

        return detected_books

    except Exception as e:
        print(f"An error occurred during detection and OCR: {e}")
        return []


def find_and_highlight_book(image_path, detected_books, search_title, output_path="highlighted_output.jpg"):
    """
    Highlights the searched book if found and saves the result.
    """
    image = cv2.imread(image_path)
    highlighted = False

    for book in detected_books:
        x1, y1, x2, y2 = book["bbox"]
        text = book["text"]

        # Use fuzzy matching for better accuracy
        similarity = fuzz.partial_ratio(search_title.lower(), text.lower())

        if similarity > 70:  # threshold, can be tuned
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, text[:20], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            highlighted = True

    cv2.imwrite(output_path, image)

    if highlighted:
        print(f"✅ Book '{search_title}' found and highlighted. Saved as {output_path}")
    else:
        print(f"❌ Book '{search_title}' not found. Output saved as {output_path} anyway.")


if __name__ == "__main__":
    # Paths
    image_file = "bookshelf_images/unseen7.jpeg"
    model_file = "runs/detect/train/weights/best.pt"

    print("Step 1: Detecting books and extracting titles...")
    book_info = detect_and_read_books(image_file, model_file)

    if not book_info:
        print("No books detected or OCR failed. Exiting.")
        sys.exit()

    print("\n--- Extracted Book Titles ---")
    for idx, info in enumerate(book_info, start=1):
        print(f"{idx}. {info['text']}")

    search_query = input("\nStep 2: Enter the name of the book you are looking for: ")

    print("\nStep 3: Searching and highlighting...")
    find_and_highlight_book(image_file, book_info, search_query)
