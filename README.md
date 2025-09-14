# 📚 Book Locator using OCR and CV on Book Spines  

This project helps **locate books in a library or bookshelf image** by combining **Computer Vision (YOLO)** for book spine detection and **OCR (Tesseract)** for extracting text from detected regions.  
A user can then search for a book title/author, and the system highlights the matching book in the image.  

---

## 🔍 Workflow  
1. **Dataset & Training**  
   - Book spine detection model trained on a dataset (e.g., via **Roboflow**) using **YOLOv8**.  
   - Model detects bounding boxes of books in a shelf image.  

2. **Detection & OCR**  
   - YOLO detects book spines.  
   - Extracted regions are passed through **PyTesseract** to recognize text (titles/authors).  

3. **Search**  
   - All detected book titles are displayed.  
   - User enters a **title/author to search**.  
   - Fuzzy string matching improves accuracy even if OCR has minor errors.  

4. **Highlight & Save**  
   - Matching book spine is highlighted with a bounding box.  
   - Final image is **saved** with highlighted results.  

---

## 🛠 Tech Stack  
- **YOLOv8** – Object Detection (Book spines)  
- **PyTesseract OCR** – Text Extraction  
- **OpenCV** – Image processing & visualization  
- **Python** – Core logic, fuzzy matching for title search  

---

## 🚀 Vision & Future Scope  
This project is a **prototype** but can scale into a **global book-finding system** for libraries, bookstores, and archives.  

- ✅ **Smart Cameras in Libraries** – Real-time book location guidance  
- ✅ **AR Glasses** – Overlay arrows highlighting the target book on shelves  
- ✅ **Laser/Projector Systems** – Physically point to the right book  
- ✅ **Robotics** – Automated retrieval from shelves  
- ✅ **Cloud Deployment** – Centralized service accessible via mobile/web apps  

---

## 🔮 Future Improvements  
- Improve OCR accuracy with **deep learning OCR models** (EAST, CRNN)  
- **Multilingual support** (English, Hindi, Spanish, etc.)  
- Handle **partially visible / tilted spines**  
- Auto-update **book catalog** by integrating with library DBs

## Limitations Faced
- Limited computational power for increased accuracy as this project was done on a laptop without GPU.

## Screenshots
### Detect books
![detected_books](https://github.com/user-attachments/assets/87d0fcf0-eeaa-44f7-817d-e673872f45ff)

### Terminal
![highlighted_output](https://github.com/user-attachments/assets/bd9377fb-661e-4f5b-a265-cce197c38d77)


### Locate the book
![highlighted_output](https://github.com/user-attachments/assets/9b3f2f9f-b16d-40d0-82a9-b609df6a89d5)



---

