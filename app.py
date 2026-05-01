from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from vehicle_counter import CountConfig, RoundaboutVehicleCounter

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "weights" / "best.pt"
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="YOLO Vehicle Counter",
    page_icon="🚗",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_first_frame(video_bytes: bytes, suffix: str) -> Tuple[Image.Image | None, Tuple[int, int]]:
    """Read first frame from uploaded video bytes for line preview."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = Path(tmp.name)

    cap = cv2.VideoCapture(str(tmp_path))
    ok, frame = cap.read()
    cap.release()
    tmp_path.unlink(missing_ok=True)

    if not ok or frame is None:
        return None, (0, 0)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    return Image.fromarray(frame_rgb), (w, h)


@st.cache_resource(show_spinner=False)
def load_counter_cached(model_path: str, model_mtime: float) -> RoundaboutVehicleCounter:
    """Load YOLO once per model path/modified time."""
    del model_mtime  # used only as cache key
    return RoundaboutVehicleCounter(model_path)


def load_counter(model_path: Path) -> RoundaboutVehicleCounter:
    return load_counter_cached(str(model_path), float(model_path.stat().st_mtime))


def draw_line_preview(image: Image.Image, line_norm: Tuple[float, float, float, float]) -> Image.Image:
    frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = line_norm
    p1 = (int(x1 * w), int(y1 * h))
    p2 = (int(x2 * w), int(y2 * h))
    cv2.line(frame, p1, p2, (0, 255, 255), 4)
    cv2.circle(frame, p1, 8, (0, 255, 255), -1)
    cv2.circle(frame, p2, 8, (0, 255, 255), -1)
    cv2.putText(
        frame,
        "COUNTING LINE",
        (p1[0] + 10, max(30, p1[1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def save_uploaded_file(uploaded_file, target_dir: Path, name: str | None = None) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = name or uploaded_file.name
    safe_name = Path(filename).name.replace(" ", "_")
    path = target_dir / safe_name
    with path.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def make_zip(paths: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            path = Path(path)
            if path.exists():
                zf.write(path, arcname=path.name)
    buffer.seek(0)
    return buffer.getvalue()


def resize_pil_keep_width(image: Image.Image, max_width: int = 900) -> Image.Image:
    image = image.convert("RGB")
    if image.width <= max_width:
        return image
    scale = max_width / float(image.width)
    new_size = (max_width, int(round(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def make_before_after_gif(before: Image.Image, after: Image.Image, max_width: int = 900) -> bytes:
    """Create a lightweight animated before/after reveal GIF without extra packages."""
    before = resize_pil_keep_width(before, max_width=max_width)
    after = after.convert("RGB").resize(before.size, Image.Resampling.LANCZOS)

    width, height = before.size
    frames: list[Image.Image] = []
    font = ImageFont.load_default()

    reveal_positions = list(np.linspace(0, width, 18).astype(int))
    reveal_positions += list(np.linspace(width, 0, 18).astype(int))

    for cut_x in reveal_positions:
        frame = before.copy()
        if cut_x > 0:
            after_crop = after.crop((0, 0, cut_x, height))
            frame.paste(after_crop, (0, 0))

        draw = ImageDraw.Draw(frame, "RGBA")
        # Header labels
        draw.rectangle((10, 10, 104, 38), fill=(0, 0, 0, 135))
        draw.text((18, 18), "AFTER", fill=(255, 255, 255, 255), font=font)
        draw.rectangle((width - 112, 10, width - 10, 38), fill=(0, 0, 0, 135))
        draw.text((width - 96, 18), "BEFORE", fill=(255, 255, 255, 255), font=font)

        # Moving reveal line
        line_x = max(0, min(width - 1, cut_x))
        draw.line((line_x, 0, line_x, height), fill=(255, 255, 0, 255), width=4)
        frames.append(frame)

    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=85,
        loop=0,
        optimize=True,
    )
    buffer.seek(0)
    return buffer.getvalue()


def model_selector() -> Path | None:
    st.sidebar.header("1. Model")
    st.sidebar.caption("Default path: `weights/best.pt`")

    model_upload = st.sidebar.file_uploader(
        "Upload best.pt if it is not already in weights/",
        type=["pt"],
        help="For free cloud deploy, uploading the model in the UI is often easiest if the .pt file is large.",
    )

    if model_upload is not None:
        model_path = save_uploaded_file(model_upload, OUTPUT_DIR / "uploaded_models", "best.pt")
        st.sidebar.success(f"Using uploaded model: {model_path.name}")
        return model_path

    if DEFAULT_MODEL_PATH.exists():
        st.sidebar.success("Found weights/best.pt")
        return DEFAULT_MODEL_PATH

    st.sidebar.warning("No model found yet. Put `best.pt` in `weights/` or upload it here.")
    return None


def parse_manual_class_ids(text: str) -> list[int]:
    if not text.strip():
        return []
    ids: list[int] = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        ids.append(int(token))
    return sorted(set(ids))


def resolve_allowed_class_ids(
    counter: RoundaboutVehicleCounter,
    class_filter_mode: str,
    manual_class_ids: str,
) -> Optional[list[int]]:
    if class_filter_mode.startswith("All"):
        return None
    if class_filter_mode.startswith("Auto"):
        ids = counter.auto_vehicle_class_ids()
        if not ids:
            st.warning(
                "Auto vehicle-name filter did not find vehicle class names in this model. "
                "I will use all model classes instead."
            )
            return None
        st.info(f"Auto vehicle class IDs used: {ids}")
        return ids
    return parse_manual_class_ids(manual_class_ids)


def common_sidebar_controls() -> tuple[float, float, int, str, str]:
    st.sidebar.header("2. Detection")
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    iou = st.sidebar.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
    imgsz = st.sidebar.select_slider("Image size", options=[416, 512, 640, 768, 960, 1280], value=960)

    st.sidebar.header("3. Class filter")
    class_filter_mode = st.sidebar.radio(
        "Count which classes?",
        [
            "All model classes (best for custom vehicle model)",
            "Auto vehicle-name filter",
            "Manual class IDs",
        ],
        index=0,
        help="For your custom vehicle model, usually choose All. For COCO models, Auto vehicle-name filter counts car/bus/truck/etc.",
    )
    manual_class_ids = st.sidebar.text_input(
        "Manual class IDs, comma separated",
        value="",
        help="Example: 0,2,3. Used only when Manual class IDs is selected.",
    )
    return conf, iou, imgsz, class_filter_mode, manual_class_ids


def show_model_classes(counter: RoundaboutVehicleCounter) -> None:
    with st.expander("Model class IDs"):
        st.dataframe(pd.DataFrame(counter.model_class_table()), use_container_width=True, hide_index=True)


def run_image_counter(
    model_path: Optional[Path],
    conf: float,
    iou: float,
    imgsz: int,
    class_filter_mode: str,
    manual_class_ids: str,
) -> None:
    st.subheader("🖼️ Image upload + vehicle counting")
    st.markdown(
        "Upload one traffic image. The app will detect vehicles, count them by class, draw boxes, "
        "and show an animated before/after comparison."
    )

    image_upload = st.file_uploader(
        "Upload input image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )

    if image_upload is None:
        st.info("Upload an image to begin.")
        return

    image_bytes = image_upload.getvalue()
    before_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.caption("Original image")
        st.image(before_image, use_container_width=True)
    with col2:
        st.caption("Image information")
        st.write(f"**File:** `{image_upload.name}`")
        st.write(f"**Size:** {before_image.width} × {before_image.height}")

    run = st.button("🚀 Detect and count vehicles in image", type="primary", use_container_width=True)
    if not run:
        return

    if model_path is None or not Path(model_path).exists():
        st.error("Missing model file. Please upload `best.pt` in the sidebar or place it at `weights/best.pt`.")
        return

    work_dir = OUTPUT_DIR / "runs"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / Path(image_upload.name).name.replace(" ", "_")
    input_path.write_bytes(image_bytes)

    try:
        with st.spinner("Running YOLO detection on image..."):
            counter = load_counter(model_path)
            allowed_class_ids = resolve_allowed_class_ids(counter, class_filter_mode, manual_class_ids)
            result = counter.process_image(
                input_image_path=input_path,
                output_dir=work_dir,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                allowed_class_ids=allowed_class_ids,
            )
    except Exception as exc:
        st.exception(exc)
        st.error("Image processing failed. Check model path, image format, class IDs, and dependencies.")
        return

    st.success("Image counting completed.")
    show_model_classes(counter)

    m1, m2 = st.columns(2)
    m1.metric("Total vehicles / objects counted", result.total_count)
    m2.metric("Elapsed seconds", result.elapsed_seconds)

    output_image_path = Path(result.output_image_path)
    csv_path = Path(result.csv_path)
    json_path = Path(result.json_path)
    after_image = Image.open(output_image_path).convert("RGB")

    st.subheader("Before / after animated comparison")
    gif_bytes = make_before_after_gif(before_image, after_image)
    st.image(gif_bytes, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Before")
        st.image(before_image, use_container_width=True)
    with c2:
        st.subheader("After: detections + count")
        st.image(after_image, use_container_width=True)

    st.subheader("Per-class count")
    if result.per_class_count:
        df_counts = pd.DataFrame([{"class_name": k, "count": v} for k, v in result.per_class_count.items()])
    else:
        df_counts = pd.DataFrame(columns=["class_name", "count"])
    st.dataframe(df_counts, use_container_width=True, hide_index=True)

    st.subheader("Detection table")
    df_det = pd.DataFrame(result.detections)
    st.dataframe(df_det, use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Download annotated image",
            data=output_image_path.read_bytes(),
            file_name=output_image_path.name,
            mime="image/jpeg",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download before/after GIF",
            data=gif_bytes,
            file_name="before_after_detection.gif",
            mime="image/gif",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "Download CSV",
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
            use_container_width=True,
        )
    with d4:
        zip_bytes = make_zip([output_image_path, csv_path, json_path])
        st.download_button(
            "Download outputs ZIP",
            data=zip_bytes,
            file_name="image_vehicle_counting_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with st.expander("JSON summary"):
        st.json(json.loads(json_path.read_text(encoding="utf-8")))


def run_video_counter(
    model_path: Optional[Path],
    conf: float,
    iou: float,
    imgsz: int,
    class_filter_mode: str,
    manual_class_ids: str,
) -> None:
    st.subheader("🎥 Video upload + line-crossing vehicle counting")
    st.markdown(
        "Upload a traffic video, choose a counting line, then run YOLO tracking. "
        "A vehicle is counted once when its tracked center crosses the line."
    )

    st.sidebar.header("4. Video tracking")
    tracker = st.sidebar.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"], index=0)
    direction = st.sidebar.selectbox(
        "Counting direction",
        ["both", "positive_to_negative", "negative_to_positive"],
        index=0,
        help="Use 'both' for simplest demo. Use one direction if you only want entry or exit counting.",
    )
    resize_width = st.sidebar.select_slider(
        "Resize video width before inference",
        options=[0, 640, 800, 960, 1280],
        value=960,
        help="Lower width runs faster on free CPU. 0 keeps original size.",
    )
    max_frames = st.sidebar.number_input(
        "Max frames to process (0 = full video)",
        min_value=0,
        max_value=200000,
        value=0,
        step=100,
        help="Use a small value like 300 for fast demo on free cloud CPU.",
    )

    st.sidebar.header("5. Counting line")
    st.sidebar.caption("Coordinates are normalized from 0 to 1, so they work for videos of different sizes.")
    x1 = st.sidebar.slider("x1", 0.0, 1.0, 0.05, 0.01)
    y1 = st.sidebar.slider("y1", 0.0, 1.0, 0.55, 0.01)
    x2 = st.sidebar.slider("x2", 0.0, 1.0, 0.95, 0.01)
    y2 = st.sidebar.slider("y2", 0.0, 1.0, 0.55, 0.01)
    line_norm = (x1, y1, x2, y2)

    video_upload = st.file_uploader(
        "Upload input video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_upload",
    )

    if video_upload is None:
        st.info("Upload a video to begin. For free cloud deployment, use a short demo video first.")
        return

    video_bytes = video_upload.getvalue()
    first_frame, size = read_first_frame(video_bytes, Path(video_upload.name).suffix or ".mp4")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Input video")
        st.video(video_bytes)
    with col2:
        st.subheader("Counting line preview")
        if first_frame is not None:
            st.image(draw_line_preview(first_frame, line_norm), use_container_width=True)
            st.caption(f"Original first-frame size: {size[0]} × {size[1]}")
        else:
            st.warning("Could not preview first frame, but you can still try processing.")

    run = st.button("🚀 Run video vehicle counting", type="primary", use_container_width=True)
    if not run:
        return

    if model_path is None or not Path(model_path).exists():
        st.error("Missing model file. Please upload `best.pt` in the sidebar or place it at `weights/best.pt`.")
        return

    work_dir = OUTPUT_DIR / "runs"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / Path(video_upload.name).name.replace(" ", "_")
    input_path.write_bytes(video_bytes)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(value: float, message: str) -> None:
        progress_bar.progress(max(0.0, min(1.0, float(value))))
        status_text.write(message)

    try:
        with st.spinner("Running YOLO tracking and line-crossing count..."):
            counter = load_counter(model_path)
            allowed_class_ids = resolve_allowed_class_ids(counter, class_filter_mode, manual_class_ids)
            config = CountConfig(
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                tracker=tracker,
                direction=direction,
                resize_width=resize_width,
                max_frames=int(max_frames),
                draw_track_history=True,
                allowed_class_ids=allowed_class_ids,
            )
            result = counter.process_video(
                input_video_path=input_path,
                output_dir=work_dir,
                line_norm=line_norm,
                config=config,
                progress_callback=update_progress,
            )
    except Exception as exc:
        st.exception(exc)
        st.error("Video processing failed. Check model path, video format, class IDs, and dependencies.")
        return

    st.success("Video counting completed.")
    show_model_classes(counter)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total vehicles counted", result.total_count)
    m2.metric("Processed frames", result.processed_frames)
    m3.metric("Elapsed seconds", result.elapsed_seconds)

    st.subheader("Per-class count")
    if result.per_class_count:
        df_counts = pd.DataFrame([{"class_name": k, "count": v} for k, v in result.per_class_count.items()])
    else:
        df_counts = pd.DataFrame(columns=["class_name", "count"])
    st.dataframe(df_counts, use_container_width=True, hide_index=True)

    output_video_path = Path(result.output_video_path)
    csv_path = Path(result.csv_path)
    json_path = Path(result.json_path)

    st.subheader("Annotated output video")
    if output_video_path.exists():
        st.video(output_video_path.read_bytes())

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download annotated video",
            data=output_video_path.read_bytes(),
            file_name=output_video_path.name,
            mime="video/mp4",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download CSV events",
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
            use_container_width=True,
        )
    with d3:
        zip_bytes = make_zip([output_video_path, csv_path, json_path])
        st.download_button(
            "Download all outputs ZIP",
            data=zip_bytes,
            file_name="video_vehicle_counting_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with st.expander("JSON summary"):
        st.json(json.loads(json_path.read_text(encoding="utf-8")))


def main() -> None:
    st.title("🚗 YOLO Vehicle Counter: Image + Video")
    st.markdown(
        "This app supports **image upload** for direct vehicle/object counting and **video upload** for line-crossing counting. "
        "Use your trained `best.pt` model from `weights/best.pt` or upload it in the sidebar."
    )

    model_path = model_selector()
    conf, iou, imgsz, class_filter_mode, manual_class_ids = common_sidebar_controls()

    task = st.sidebar.radio(
        "4. Input type",
        ["Image counter", "Video counter"],
        index=0,
    )

    if task == "Image counter":
        run_image_counter(model_path, conf, iou, imgsz, class_filter_mode, manual_class_ids)
    else:
        run_video_counter(model_path, conf, iou, imgsz, class_filter_mode, manual_class_ids)


if __name__ == "__main__":
    main()
