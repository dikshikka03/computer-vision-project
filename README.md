# 👁️ Face and Eye Detection using OpenCV

This project uses OpenCV and Haar Cascade classifiers to detect faces and eyes in images and live webcam feed.

---

## 🚀 Features
- Face detection in images
- Eye detection inside detected faces
- Live face & eye detection using webcam

---

## 🛠️ Technologies Used
- Python
- OpenCV (cv2)
- NumPy

---

## 📂 Project Structure
Face-Detection-Project/
│
├── detectFacesAndEyes.py
├── liveFacesAndEyes.py
├── images/
├── cascades/
└── results/

## ▶️ How to Run

### 1. Install dependencies
pip install opencv-python numpy

### 2. Run image detection
python detectFacesAndEyes.py

### 3. Run live detection
python liveFacesAndEyes.py

Press **Q** to exit webcam.
## 📸 Output

### Original Image
![Original](images/children.jpg)
![Original](images/Hillary.jpg)
![Original](images/narcos.jpg)
![Original](images/svalley.jpg)
![Original](images/Trump.jpg)

### Detected Faces & Eyes
![Detected](results/output.jpg)
- Detects faces using rectangles
- Detects eyes inside faces
- Works on both images and live webcam

- ## ⚠️ Important

- Ensure Haar cascade files are in the `cascades/` folder
- Update image path if needed

- ## 💡 Future Improvements

- Add smile detection
- Add emotion recognition
- Improve detection accuracy using deep learning

- ## 👩‍💻 Author

Dikshika
