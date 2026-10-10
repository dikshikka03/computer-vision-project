
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

    if isinstance(label_data, dict) and "images" in label_data:
        entries = label_data["images"]
    elif isinstance(label_data, dict):
        entries = [
            {"filename": filename, "faces": faces}
            for filename, faces in label_data.items()
        ]
    elif isinstance(label_data, list):
        entries = label_data
    else:
        raise ValueError("Unsupported labels.json format.")

    face_cascade = load_face_cascade()
    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        filename = entry.get("filename") or entry.get("image")
        actual_boxes = entry.get("faces", entry.get("boxes", []))

        if not filename:
            continue

        image_path = IMAGE_DIR / Path(filename).name

        if not image_path.exists():
            results.append({
                "Image": filename,
                "Actual Faces": len(actual_boxes),
                "Detected Faces": 0,
                "TP": 0,
                "FP": 0,
                "FN": len(actual_boxes),
                "Precision": 0.0,
                "Recall": 0.0,
                "F1 Score": 0.0,
                "Status": "Image not found"
            })
            total_fn += len(actual_boxes)
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

        normalized_actual = []

        for box in actual_boxes:
            if isinstance(box, dict):
                if all(key in box for key in ("x", "y", "w", "h")):
                    normalized_actual.append((
                        int(box["x"]),
                        int(box["y"]),
                        int(box["w"]),
                        int(box["h"])
                    ))
                elif all(key in box for key in ("x", "y", "width", "height")):
                    normalized_actual.append((
                        int(box["x"]),
                        int(box["y"]),
                        int(box["width"]),
                        int(box["height"])
                    ))
            elif isinstance(box, (list, tuple)) and len(box) == 4:
                normalized_actual.append(tuple(map(int, box)))

        predicted_boxes = [
            tuple(map(int, box)) for box in predicted_boxes
        ]

        tp, fp, fn = match_boxes(
            predicted_boxes,
            normalized_actual,
            threshold=iou_threshold
        )

        precision, recall, f1 = calculate_metrics(tp, fp, fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        results.append({
            "Image": filename,
            "Actual Faces": len(normalized_actual),
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
        total_tp,
        total_fp,
        total_fn
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
