
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


st.set_page_config(
    page_title="VisionLens | Computer Vision Studio",
    page_icon="👁️",
    layout="wide",
)

st.title("👁️ VisionLens")
st.subheader("Computer Vision Studio")
st.markdown(
    "Explore face and eye detection with "
    "Python, OpenCV, and Haar Cascade classifiers."
)
st.write(
    "An interactive computer vision application for detecting "
    "faces and eyes in images using OpenCV and Haar Cascade classifiers."
)

st.divider()


st.info(
    "Upload a JPG or PNG image to detect faces and eyes."
)


@st.cache_resource
def load_cascades():
    face_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    eye_path = (
        cv2.data.haarcascades
        + "haarcascade_eye.xml"
    )

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)

    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Unable to load Haar Cascade classifiers.")

    return face_cascade, eye_cascade


def detect_faces_and_eyes(image):
    face_cascade, eye_cascade = load_cascades()

    image_rgb = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
    )

    eye_count = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(
            image_bgr,
            (x, y),
            (x + w, y + h),
            (0, 0, 0),
            3,
        )

        face_gray = gray[y:y + h, x:x + w]
        face_color = image_bgr[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(10, 10),
        )

        eye_count += len(eyes)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2,
            )

    result_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return result_rgb, len(faces), eye_count


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    try:
        original = Image.open(uploaded_file).convert("RGB")

        st.subheader("Original Image")
        st.image(original, use_container_width=True)

        if st.button("Run Detection", type="primary"):
            with st.spinner("Detecting faces and eyes..."):
                result, face_count, eye_count = (
                    detect_faces_and_eyes(original)
                )

            st.subheader("Detection Results")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    original,
                    caption="Original",
                    use_container_width=True,
                )

            with col2:
                st.image(
                    result,
                    caption="Detected faces and eyes",
                    use_container_width=True,
                )

            metric1, metric2 = st.columns(2)
            metric1.metric("Faces Detected", face_count)
            metric2.metric("Eyes Detected", eye_count)

            if face_count == 0:
                st.warning(
                    "No faces detected. Try a clear, "
                    "front-facing image with good lighting."
                )
            else:
                st.success("Detection completed!")

            buffer = BytesIO()
            Image.fromarray(result).save(buffer, format="PNG")

            st.download_button(
                "Download Annotated Image",
                data=buffer.getvalue(),
                file_name="face_eye_detection.png",
                mime="image/png",
            )


         
    except Exception as error:
        st.error("Image processing failed.")
        st.exception(error)


st.divider()
st.caption(
    "Built with Python, OpenCV, Haar Cascades, NumPy, "
    "Pillow, and Streamlit."
)
