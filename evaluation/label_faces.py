
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
LABEL_FILE = BASE_DIR / "labels.json"

image_files = [
    "narcos.jpg",
    "results - 2.png",
    "Hillary.jpg",
    "Trump.jpg",
    "children.jpg",
]

labels = {}
current_index = 0
boxes = []
start_x = start_y = None
scale = 1.0
photo = None


def load_labels():
    if LABEL_FILE.exists():
        try:
            data = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
            for item in data.get("images", []):
                labels[item["filename"]] = item.get("faces", [])
        except (json.JSONDecodeError, KeyError):
            pass

    for filename in image_files:
        labels.setdefault(filename, [])


def save_labels():
    data = {
        "images": [
            {"filename": filename, "faces": labels[filename]}
            for filename in image_files
        ]
    }
    LABEL_FILE.write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def show_image():
    global photo, boxes, scale

    filename = image_files[current_index]
    path = IMAGE_DIR / filename

    if not path.exists():
        messagebox.showerror("Missing image", f"Cannot find:\n{path}")
        return

    image = Image.open(path).convert("RGB")
    original_width, original_height = image.size

    max_width, max_height = 850, 520
    scale = min(
        max_width / original_width,
        max_height / original_height,
        1.0,
    )

    display_width = int(original_width * scale)
    display_height = int(original_height * scale)
    image = image.resize((display_width, display_height))

    photo = ImageTk.PhotoImage(image)
    canvas.config(width=display_width, height=display_height)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=photo)

    boxes = []
    for x, y, w, h in labels[filename]:
        rectangle = canvas.create_rectangle(
            x * scale, y * scale,
            (x + w) * scale, (y + h) * scale,
            outline="lime", width=2,
        )
        boxes.append((rectangle, [x, y, w, h]))

    status.config(
        text=f"Image {current_index + 1}/{len(image_files)}: "
             f"{filename} | Faces labelled: {len(boxes)}"
    )


def mouse_down(event):
    global start_x, start_y
    start_x, start_y = event.x, event.y


def mouse_up(event):
    if start_x is None or start_y is None:
        return

    x1, y1 = start_x, start_y
    x2, y2 = event.x, event.y

    left, top = min(x1, x2), min(y1, y2)
    right, bottom = max(x1, x2), max(y1, y2)

    if right - left < 5 or bottom - top < 5:
        return

    filename = image_files[current_index]
    original = Image.open(IMAGE_DIR / filename)
    width, height = original.size

    left = max(0, min(int(left / scale), width - 1))
    top = max(0, min(int(top / scale), height - 1))
    right = max(left + 1, min(int(right / scale), width))
    bottom = max(top + 1, min(int(bottom / scale), height))

    box = [left, top, right - left, bottom - top]
    labels[filename].append(box)

    rectangle = canvas.create_rectangle(
        left * scale, top * scale,
        right * scale, bottom * scale,
        outline="lime", width=2,
    )
    boxes.append((rectangle, box))

    status.config(
        text=f"Image {current_index + 1}/{len(image_files)}: "
             f"{filename} | Faces labelled: {len(boxes)}"
    )


def undo_box():
    if not boxes:
        return

    rectangle, _ = boxes.pop()
    canvas.delete(rectangle)
    filename = image_files[current_index]
    labels[filename].pop()

    status.config(
        text=f"Image {current_index + 1}/{len(image_files)}: "
             f"{filename} | Faces labelled: {len(boxes)}"
    )


def next_image():
    global current_index
    save_labels()

    if current_index < len(image_files) - 1:
        current_index += 1
        show_image()
    else:
        messagebox.showinfo(
            "Finished",
            "All images have been reviewed and labels.json is saved."
        )


def previous_image():
    global current_index
    save_labels()

    if current_index > 0:
        current_index -= 1
        show_image()


load_labels()

root = tk.Tk()
root.title("VisionLens - Face Labeling Tool")
root.geometry("1000x680")

tk.Label(
    root,
    text="Draw a rectangle around EACH visible face",
    font=("Arial", 14, "bold"),
).pack(pady=8)

status = tk.Label(root, text="", font=("Arial", 10))
status.pack(pady=4)

canvas = tk.Canvas(root, bg="gray")
canvas.pack(padx=10, pady=5)

canvas.bind("<ButtonPress-1>", mouse_down)
canvas.bind("<ButtonRelease-1>", mouse_up)

controls = tk.Frame(root)
controls.pack(pady=12)

tk.Button(
    controls, text="Undo Last Box",
    command=undo_box, width=15
).pack(side="left", padx=5)

tk.Button(
    controls, text="Previous Image",
    command=previous_image, width=15
).pack(side="left", padx=5)

tk.Button(
    controls, text="Save & Next",
    command=next_image, width=15
).pack(side="left", padx=5)

tk.Label(
    root,
    text="Include every clearly visible face. Draw a tight box around each face.",
).pack(pady=5)

show_image()
root.mainloop()


