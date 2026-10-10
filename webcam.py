
import time

import cv2


def main():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    if face_cascade.empty() or eye_cascade.empty():
        print("Could not load the face or eye detection model.")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not access the webcam. Check camera permissions.")
        return

    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a frame from the webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            eye_count = 0

            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                face_gray = gray[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(
                    face_gray,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(10, 10)
                )

                eye_count += len(eyes)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        frame,
                        (x + ex, y + ey),
                        (x + ex + ew, y + ey + eh),
                        (0, 165, 255),
                        2
                    )

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            fps = 1 / elapsed if elapsed > 0 else 0

            cv2.putText(
                frame,
                f"Faces: {len(faces)} | Eyes: {eye_count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Press Q to quit",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("VisionLens - Live Webcam Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
