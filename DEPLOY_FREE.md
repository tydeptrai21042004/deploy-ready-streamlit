# Free deployment guide

## Best free path for this project

Use **Hugging Face Spaces with Streamlit SDK** or **Streamlit Community Cloud**.

Because YOLO video inference is heavy, the simplest reliable free workflow is:

1. deploy the app code only;
2. upload `best.pt` from the sidebar when running the app;
3. use short demo videos for cloud CPU;
4. for full videos, run locally or on a GPU machine.

---

## Hugging Face Spaces

1. Create account on Hugging Face.
2. Create new Space.
3. SDK: Streamlit.
4. Upload all files from this folder.
5. Wait for build.
6. Open app.
7. Upload `best.pt` and video.

Optional permanent model setup:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes weights/best.pt
git commit -m "Add YOLO model"
git push
```

If doing this, remove or edit the `weights/*.pt` rule in `.gitignore`.

---

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Go to Streamlit Community Cloud.
3. New app → choose repo.
4. Main file path: `app.py`.
5. Deploy.
6. Upload `best.pt` from sidebar.

Note: GitHub has file-size restrictions. If `best.pt` is large, do not commit it directly.

---

## Local Windows command

```powershell
cd roundabout-yolov11-streamlit-deploy
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Local Linux/macOS command

```bash
cd roundabout-yolov11-streamlit-deploy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
