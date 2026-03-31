import cv2
import sys
import os
import argparse

# 1. Setup Argument Parser for CLI input
def main():
    parser = argparse.ArgumentParser(description="CLI Face and Eye Detector")
    parser.add_argument("--input", required=True, help="Path to the input image file")
    parser.add_argument("--output", default="output_detected.jpg", help="Path to save the processed image")
    args = parser.parse_args()

    # 2. Use robust paths for Cascades (Points to OpenCV's internal library files)
    cascade_face = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade_eye = cv2.data.haarcascades + 'haarcascade_eye.xml'

    face_classifier = cv2.CascadeClassifier(cascade_face)
    eye_classifier = cv2.CascadeClassifier(cascade_eye)

    # 3. Load the image provided via CLI
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found.")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print("Error: Could not decode image.")
        sys.exit(1)

    # 4. Processing Logic (Grayscale + Detection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        print("No faces detected.")
    else:
        print(f"Detected {len(faces)} face(s). Processing...")

    for (x, y, w, h) in faces:
        # Draw Face Rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 3)

        # Region of Interest (ROI) for eyes within the face
        area_gray = gray[y:y+h, x:x+w]
        area_original = image[y:y+h, x:x+w]

        eyes = eye_classifier.detectMultiScale(area_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw Eye Rectangles
            cv2.rectangle(area_original, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # 5. Save the output (Crucial for CLI environments)
    cv2.imwrite(args.output, image)
    print(f"Success! Processed image saved as: {args.output}")

if __name__ == "__main__":
    main()
