# Free deployment guide — fixed Streamlit Cloud version

## Important fix

This version intentionally has **no `packages.txt`**.

Your previous Streamlit Cloud build failed during the Linux `apt-get` step because `packages.txt` requested packages such as `ffmpeg`, `libgl1`, and `libglib2.0-0`. On the current Streamlit Cloud base image, those packages can pull conflicting Debian versions.

The app now relies on Python wheels instead:

```text
opencv-python-headless
ultralytics
streamlit
```

So the cloud build should go directly to Python dependency installation.

---

## Deploy on Streamlit Community Cloud

1. Unzip this folder.
2. Put the files into your GitHub repository.
3. Make sure `packages.txt` is **not** in the repository.
4. Push to GitHub.
5. In Streamlit Cloud, reboot/redeploy the app.
6. Upload `best.pt` in the sidebar, or place it at:

```text
weights/best.pt
```

For large `.pt` files, uploading in the sidebar is easier than committing to GitHub.

---

## Local Windows command

```powershell
cd roundabout-yolov11-streamlit-deploy-fixed
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Local Linux/macOS command

```bash
cd roundabout-yolov11-streamlit-deploy-fixed
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## If video output does not preview in browser

The counter still saves the output video. Download it from the app and open it locally.
Some free cloud environments may have limited MP4 codec support.
