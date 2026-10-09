# 👁️ VisionLens — Computer Vision Studio

An interactive computer vision application built with Python, OpenCV, and Streamlit to detect faces and eyes in uploaded images using Haar Cascade classifiers.
## 🌐 Live Demo

Try VisionLens here: **[Open VisionLens](PASTE_YOUR_STREAMLIT_URL_HERE)**
## 🚀 Features

- Face detection in images
- Eye detection within detected face regions
- Real-time visual feedback through annotated images
- Face and eye detection counts
- Original image and detection output comparison
- Downloadable annotated results
- Interactive web interface

## 🛠️ Technologies Used

- **Python** — Core programming language
- **OpenCV** — Image processing and object detection
- **Haar Cascade Classifiers** — Face and eye detection
- **NumPy** — Image array processing
- **Pillow (PIL)** — Image handling
- **Streamlit** — Interactive web application

## ⚙️ Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/dikshikka03/computer-vision-project.git
   ```

2. Navigate to the project directory:

   ```bash
   cd computer-vision-project
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

## 🧠 How It Works

1. The user uploads an image.
2. OpenCV converts the image into grayscale.
3. Haar Cascade classifiers detect faces.
4. Eye detection runs within the detected face regions.
5. The application displays the annotated image and detection counts.
6. The user can download the processed image.

## ⚠️ Limitations

Detection performance can vary depending on lighting, face orientation, image quality, and occlusion. Haar Cascade classifiers may produce false positives or miss faces.

## 👩‍💻 Author

**Dikshika**

GitHub: [@dikshikka03](https://github.com/dikshikka03)
