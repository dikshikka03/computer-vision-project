from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import json
import time
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="VisionLens | Computer Vision Studio",
    page_icon="👁️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = BASE_DIR / "evaluation"
IMAGE_DIR = EVALUATION_DIR / "images"
LABEL_FILE = EVALUATION_DIR / "labels.json"

st.markdown(
    """
    <style>
   
.main {

/* Main application background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #f4f6fc !important;
}

/* Main content panel */
[data-testid="stMainBlockContainer"] {
    background-color: #f4f6fc !important;
    padding: 2rem 2.5rem !important;
}

/* Selected tab */

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #4338ca !important;
    border-bottom: 3px solid #4f46e5 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #4f46e5 !important;
}

.stTabs [data-baseweb="tab-border"] {
    background-color: #dbe3f0 !important;
}


/* Metrics */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #dbe3f0 !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}

/* Primary buttons */
.stButton button[kind="primary"] {
    background-color: #4f46e5 !important;
    color: white !important;
    border-color: #4f46e5 !important;
}

/* Section headings */
h1, h2, h3 {
    color: #172554 !important;
}


    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>VisionLens | Computer Vision Studio</h1>
        <p>Face and eye detection powered by OpenCV Haar Cascade classifiers.</p>
    </div>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_face_cascade():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError("Unable to load the face detection model.")
    return cascade


@st.cache_resource
def load_eye_cascade():
    path = cv2.data.haarcascades + "haarcascade_eye.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError("Unable to load the eye detection model.")
    return cascade


def detect_faces_and_eyes(
    image,
    face_scale_factor=1.1,
    face_min_neighbors=5,
    face_min_size=30,
    eye_scale_factor=1.1,
    eye_min_neighbors=6,
    eye_min_size=10
):
    start_time = time.perf_counter()

    face_cascade = load_face_cascade()
    eye_cascade = load_eye_cascade()

    image_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    face_boxes = face_cascade.detectMultiScale(
        gray,
        scaleFactor=face_scale_factor,
        minNeighbors=face_min_neighbors,
        minSize=(face_min_size, face_min_size)
    )

    annotated = image_array.copy()
    total_eyes = 0

    for (x, y, w, h) in face_boxes:
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=eye_scale_factor,
            minNeighbors=eye_min_neighbors,
            minSize=(eye_min_size, eye_min_size)
        )

        total_eyes += len(eyes)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                annotated,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (0, 165, 255),
                2
            )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return (
        Image.fromarray(annotated),
        len(face_boxes),
        total_eyes,
        elapsed_ms,
        face_boxes
    )


def calculate_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_boxes(predicted_boxes, actual_boxes, threshold=0.5):
    matches = []
    used_actual = set()

    for predicted in predicted_boxes:
        best_iou = 0.0
        best_index = None

        for index, actual in enumerate(actual_boxes):
            if index in used_actual:
                continue

            iou = calculate_iou(predicted, actual)

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= threshold:
            matches.append(best_index)
            used_actual.add(best_index)

    true_positives = len(matches)
    false_positives = len(predicted_boxes) - true_positives
    false_negatives = len(actual_boxes) - true_positives

    return true_positives, false_positives, false_negatives


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1



def evaluate_dataset(
    face_scale_factor=1.1,
    face_min_neighbors=5,
    face_min_size=30,
    iou_threshold=0.5
):
    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Ground-truth labels file not found: {LABEL_FILE}"
        )

    with open(LABEL_FILE, "r", encoding="utf-8") as file:
        label_data = json.load(file)

    entries = label_data.get("images", [])

    face_cascade = load_face_cascade()
    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for entry in entries:
        filename = entry["filename"]
        actual_boxes = entry.get("faces", [])
        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            st.warning(f"Image not found: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        predicted_boxes = face_cascade.detectMultiScale(
            gray,
            scaleFactor=face_scale_factor,
            minNeighbors=face_min_neighbors,
            minSize=(face_min_size, face_min_size)
        )

        predicted_boxes = [
            tuple(map(int, box)) for box in predicted_boxes
        ]
        actual_boxes = [
            tuple(map(int, box)) for box in actual_boxes
        ]

        tp, fp, fn = match_boxes(
            predicted_boxes,
            actual_boxes,
            threshold=iou_threshold
        )

        precision, recall, f1 = calculate_metrics(tp, fp, fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        results.append({
            "Image": filename,
            "Actual Faces": len(actual_boxes),
            "Detected Faces": len(predicted_boxes),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Status": "Evaluated"
        })

    precision, recall, f1 = calculate_metrics(
        total_tp, total_fp, total_fn
    )

    overall = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn
    }

    return results, overall


class WebcamProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.face_count = 0
        self.eye_count = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        eye_count = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            face_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(10, 10)
            )

            eye_count += len(eyes)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    image,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (0, 165, 255),
                    2
                )

        self.face_count = len(faces)
        self.eye_count = eye_count

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


with st.sidebar:
    st.header("Detection Settings")

    st.subheader("Face Detection")

    face_scale_factor = st.slider(
        "Face scale factor",
        min_value=1.05,
        max_value=1.50,
        value=1.10,
        step=0.05
    )

    face_min_neighbors = st.slider(
        "Face minimum neighbors",
        min_value=3,
        max_value=10,
        value=5
    )

    face_min_size = st.slider(
        "Minimum face size",
        min_value=20,
        max_value=100,
        value=30,
        step=5
    )

    st.subheader("Eye Detection")

    eye_scale_factor = st.slider(
        "Eye scale factor",
        min_value=1.05,
        max_value=1.50,
        value=1.10,
        step=0.05
    )

    eye_min_neighbors = st.slider(
        "Eye minimum neighbors",
        min_value=3,
        max_value=10,
        value=6
    )

    eye_min_size = st.slider(
        "Minimum eye size",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )

    st.divider()

    iou_threshold = st.slider(
        "Evaluation IoU threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05
    )



image_tab, webcam_tab, evaluation_tab = st.tabs([
    "Image Detection",
    "Live Webcam",
    "Model Evaluation"
])

with image_tab:
    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image to analyze",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        try:
            original = Image.open(uploaded_file).convert("RGB")
            image_width, image_height = original.size

            st.subheader("Original Image")

            _, image_column, _ = st.columns([1, 2, 1])

            with image_column:
                st.image(
                    original,
                    caption="Uploaded Image",
                    width=300
                )

            if st.button(
                "Run Detection",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Detecting faces and eyes..."):
                    annotated, face_count, eye_count, elapsed_ms, face_boxes = (
                        detect_faces_and_eyes(
                            original,
                            face_scale_factor,
                            face_min_neighbors,
                            face_min_size,
                            eye_scale_factor,
                            eye_min_neighbors,
                            eye_min_size
                        )
                    )

                    st.session_state["visionlens_result"] = {
                        "annotated": annotated,
                        "face_count": face_count,
                        "eye_count": eye_count,
                        "elapsed_ms": elapsed_ms,
                        "width": image_width,
                        "height": image_height,
                        "filename": uploaded_file.name
                    }

            if "visionlens_result" in st.session_state:
                result = st.session_state["visionlens_result"]

                if result["filename"] == uploaded_file.name:
                    st.subheader("Detection Results")

                    original_column, result_column = st.columns(2)

                    with original_column:
                        st.image(
                            original,
                            caption="Original",
                            use_container_width=True
                        )

                    with result_column:
                        st.image(
                            result["annotated"],
                            caption="Detected Faces and Eyes",
                            use_container_width=True
                        )

                    st.subheader("Performance Dashboard")

                    metric1, metric2, metric3, metric4 = st.columns(4)

                    metric1.metric(
                        "Faces Detected",
                        result["face_count"]
                    )

                    metric2.metric(
                        "Eyes Detected",
                        result["eye_count"]
                    )

                    metric3.metric(
                        "Processing Time",
                        f"{result['elapsed_ms']:.2f} ms"
                    )

                    metric4.metric(
                        "Image Dimensions",
                        f"{result['width']} × {result['height']}"
                    )

                    output = BytesIO()
                    result["annotated"].save(output, format="PNG")

                    st.download_button(
                        "Download Annotated Image",
                        data=output.getvalue(),
                        file_name="visionlens_result.png",
                        mime="image/png",
                        use_container_width=True
                    )

        except Exception as error:
            st.error("Unable to process this image.")
            st.exception(error)

    else:
        st.info("Upload an image to start face and eye detection.")

    st.divider()


with evaluation_tab:
    st.subheader("Model Evaluation Dashboard")

    st.write(
        "Evaluate face detection against manually labelled images "
        "using Intersection over Union (IoU)."
    )

    if st.button(
        "Run Model Evaluation",
        use_container_width=True
    ):
        try:
            with st.spinner("Evaluating the dataset..."):
                evaluation_results, overall = evaluate_dataset(
                    face_scale_factor=face_scale_factor,
                    face_min_neighbors=face_min_neighbors,
                    face_min_size=face_min_size,
                    iou_threshold=iou_threshold
                )

            st.session_state["visionlens_evaluation"] = {
                "results": evaluation_results,
                "overall": overall
            }

        except Exception as error:
            st.error("Model evaluation failed.")
            st.exception(error)

    if "visionlens_evaluation" in st.session_state:
        evaluation = st.session_state["visionlens_evaluation"]
        overall = evaluation["overall"]
        evaluation_results = evaluation["results"]

        st.subheader("Overall Evaluation Metrics")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Precision",
            f"{overall['Precision']:.3f}"
        )

        metric2.metric(
            "Recall",
            f"{overall['Recall']:.3f}"
        )

        metric3.metric(
            "F1 Score",
            f"{overall['F1 Score']:.3f}"
        )

        metric4, metric5, metric6 = st.columns(3)

        metric4.metric("True Positives", overall["TP"])
        metric5.metric("False Positives", overall["FP"])
        metric6.metric("False Negatives", overall["FN"])

        st.subheader("Per-Image Evaluation")

        dataframe = pd.DataFrame(evaluation_results)

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )

        chart_data = dataframe[
            ["Image", "Precision", "Recall", "F1 Score"]
        ].copy()

        chart_data = chart_data.set_index("Image")

        st.subheader("Performance Comparison")

        st.bar_chart(chart_data)

        st.caption(
            "Precision measures how many detections were correct. "
            "Recall measures how many labelled faces were detected. "
            "The F1 score combines precision and recall. "
            "Results depend on the labelled dataset and selected settings."
        )

    st.divider()

    st.caption(
        "VisionLens | Computer Vision Studio | "
        "Built with Python, Streamlit and OpenCV"
    )

    st.divider()


with webcam_tab:
    st.header("📹 Live Webcam Detection")
    st.write("Detect faces and eyes in real time using your webcam.")

    webrtc_streamer(
        key="visionlens-webcam",
        video_processor_factory=WebcamProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

st.divider()
st.caption(
    "VisionLens | Computer Vision Studio | "
    "Built with Python, Streamlit and OpenCV"
)
