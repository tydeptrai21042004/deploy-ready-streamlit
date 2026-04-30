# Fix for Streamlit Cloud apt dependency error

## Error symptom

The deploy log fails before Python packages install, with messages similar to:

```text
ffmpeg : Depends: libavcodec61
libglib2.0-0 : Depends: libffi7 but it is not installable
E: Unable to correct problems, you have held broken packages.
```

## Cause

`packages.txt` asked Streamlit Cloud to install system packages through `apt-get`.
The current Streamlit Cloud base image can mix Debian package sources, causing version conflicts for `ffmpeg`, `libglib2.0-0`, or `libgl1`.

## Fix applied

- Removed `packages.txt` completely.
- Kept video processing through Python wheels.
- Used `opencv-python-headless` in `requirements.txt`.

## What to push to GitHub

Push this folder exactly as-is. Make sure your repository does **not** contain:

```text
packages.txt
```

Then redeploy/reboot the Streamlit app.
