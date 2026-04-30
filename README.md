---
title: YOLOv11 Roundabout Vehicle Counter
emoji: 🚗
sdk: streamlit
app_file: app.py
pinned: false
---

# YOLOv11 Roundabout Vehicle Counter

A complete deployable **Streamlit** project for counting vehicles in a roundabout video using your trained YOLOv11 `best.pt` model.

The app supports:

- upload `best.pt` directly in the sidebar, or place it in `weights/best.pt`;
- upload a traffic video;
- preview and adjust the counting line;
- run YOLO tracking with ByteTrack or BoT-SORT;
- count each tracked vehicle once when it crosses the line;
- export annotated video, CSV crossing events, JSON summary, and output ZIP.

Your notebook class mapping is preserved:

| ID | Class |
|---:|---|
| 0 | car |
| 1 | cycle |
| 2 | bus |
| 3 | truck |
| 4 | van |

---

## 1. Project structure

```text
roundabout-yolov11-streamlit-deploy/
├── app.py
├── count_video.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .streamlit/
│   └── config.toml
├── vehicle_counter/
│   ├── __init__.py
│   ├── counter.py
│   └── geometry.py
├── weights/
│   └── PUT_BEST_PT_HERE.txt
├── sample_inputs/
└── outputs/
```

---

## 2. Run locally

### Step 1 — create environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 2 — install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 — add your model

Copy your trained model to:

```text
weights/best.pt
```

Or upload `best.pt` from the Streamlit sidebar when the app is running.

### Step 4 — run app

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

---

## 3. Command-line usage

You can also run without the web UI:

```bash
python count_video.py \
  --model weights/best.pt \
  --video sample_inputs/demo.mp4 \
  --output-dir outputs \
  --conf 0.25 \
  --imgsz 960 \
  --tracker bytetrack.yaml \
  --x1 0.05 --y1 0.55 --x2 0.95 --y2 0.55
```

The line coordinates are normalized, so they are always in `[0, 1]`.

Example:

```text
x1=0.05, y1=0.55, x2=0.95, y2=0.55
```

means a horizontal counting line across the video at 55% frame height.

---

## 4. Free deployment options

### Option A — Hugging Face Spaces, easiest for demo

1. Create a new Space.
2. Choose **Streamlit** as SDK.
3. Upload this project.
4. Start the app.
5. Upload `best.pt` in the sidebar at runtime.

For permanent model storage on Hugging Face, use Git LFS and remove the `weights/*.pt` ignore rule in `.gitignore`.

### Option B — Streamlit Community Cloud

1. Push this folder to GitHub.
2. Deploy the repo on Streamlit Community Cloud.
3. Use `app.py` as the entrypoint.
4. Upload `best.pt` in the sidebar.

For large `.pt` files, sidebar upload is safer than committing the model to GitHub.

### Option C — Docker

```bash
docker build -t roundabout-yolov11-counter .
docker run -p 8501:8501 -v %cd%/weights:/app/weights roundabout-yolov11-counter
```

On macOS/Linux:

```bash
docker run -p 8501:8501 -v $(pwd)/weights:/app/weights roundabout-yolov11-counter
```

---

## 5. How counting works

The logic is:

1. YOLO detects vehicle boxes.
2. The tracker assigns a persistent `track_id` to each vehicle.
3. The center of each box is tracked frame by frame.
4. If the center crosses the counting line, the app counts that `track_id` once.
5. The same `track_id` is never counted twice.

This is more correct than counting detections per frame because it avoids counting the same vehicle repeatedly.

---

## 6. Output files

After processing, the app creates:

```text
outputs/runs/*_counted_*.mp4
outputs/runs/*_counts_*.csv
outputs/runs/*_summary_*.json
```

CSV columns:

| Column | Meaning |
|---|---|
| frame | frame where crossing happened |
| track_id | tracker ID |
| class_id | YOLO class ID |
| class_name | class name |
| confidence | detection confidence |
| center_x | crossing center x |
| center_y | crossing center y |
| count_total_after_event | total count after this event |

---

## 7. Recommended demo settings for free CPU

For short demo videos:

```text
conf = 0.25
imgsz = 640 or 960
resize_width = 640 or 960
tracker = bytetrack.yaml
max_frames = 300 to 1000 for quick demo
```

For better quality on a local GPU:

```text
imgsz = 960 or 1280
resize_width = 0 or 1280
max_frames = 0
```

---

## 8. Common issues

### App says model missing

Put the model here:

```text
weights/best.pt
```

or upload `best.pt` in the sidebar.

### Video is slow on cloud

Use:

```text
imgsz = 640
resize_width = 640
max_frames = 300
```

Free CPU deployments are much slower than local GPU.

### Count is wrong

Adjust the counting line so that vehicles cross it clearly. Avoid placing the line where vehicles stop, overlap, or move parallel to the line.

### Browser cannot preview output video

The file is still downloadable. Some cloud environments have limited video codec support. Download the MP4 and open it locally.

---

## Streamlit Cloud apt dependency note

This fixed version intentionally does **not** include `packages.txt`.
The app uses `opencv-python-headless`, so Streamlit Cloud should skip the apt package step.
If a previous deployment failed while installing `ffmpeg`, `libgl1`, or `libglib2.0-0`, delete `packages.txt` from the GitHub repository and redeploy.

