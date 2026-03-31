import cv2
import argparse
import sys
import os

def main():
    # 1. Initialize Argument Parser
    parser = argparse.ArgumentParser(description="CLI Face and Eye Detection Tool")
    parser.add_argument("--input", required=True, help="Path to the input image")
    parser.add_argument("--output", default="detected_output.jpg", help="Path to save the result")
    args = parser.parse_args()

    # 2. Use robust paths for Cascades
    # Using cv2.data.haarcascades ensures it works on the evaluator's machine
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

    face_classifier = cv2.CascadeClassifier(face_cascade_path)
    eye_classifier = cv2.CascadeClassifier(eye_cascade_path)

    # 3. Load the image from CLI argument
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print("Error: Could not open or read the image.")
        sys.exit(1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Detection Logic
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        print(f"Found {len(faces)} face(s). Processing...")
        for (x, y, w, h) in faces:
            # Draw Face Rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 3)

            # Crop for eye detection
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]

            eyes = eye_classifier.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # Draw Eye Rectangle
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # 5. SAVE the image instead of showing it
        cv2.imwrite(args.output, image)
        print(f"Success! Output saved to: {args.output}")
    else:
        print("No faces detected in the provided image.")

if __name__ == "__main__":
    main()
