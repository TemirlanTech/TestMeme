c# MemeCV

MemeCV is a Python/OpenCV webcam application that uses MediaPipe Face Mesh and Hands to detect expressions and gestures, then displays a matching meme image in a second window.

## Features

- Mirrors the live webcam feed horizontally.
- Draws Face Mesh landmarks as white dots.
- Draws hand landmarks and connections in pink.
- Detects a big, open smile, an upward-looking/thinking pose, a peace sign, a thumbs-up, and a two-hand T timeout gesture.
- Requires the same detected state for 7 consecutive frames before changing the meme window.
- Press **Esc** to exit.

## Setup

Use a native Windows terminal such as PowerShell or Command Prompt. Do not run the webcam program from WSL2 because WSL2 may not expose the Windows camera device correctly.

```powershell
cd MemeCV
py -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python main.py
```

The requirements file pins MediaPipe to `0.10.21` because this version of MemeCV uses the legacy `mp.solutions.face_mesh` and `mp.solutions.hands` interfaces. MediaPipe `0.10.31` and later removed those interfaces. If you already installed a newer version, run the following in the same activated virtual environment:

```powershell
python -m pip uninstall mediapipe -y
python -m pip install -r requirements.txt
```

If the error continues, make sure the project folder does not contain a file or folder named `mediapipe.py` or `mediapipe`, since that can shadow the installed package.

To select another camera device, for example camera index 1:

```powershell
python main.py --camera 1
```

If the webcam cannot be opened, MemeCV prints a specific message explaining that it should be run from a Windows terminal rather than WSL2.

## Meme assets

Put these files in `assets/new/`:

| Detection | File |
|---|---|
| Big smile or peace sign | `109fb257daabe2f3db63bd7bc1944934.jpg` |
| Looking up / thinking or default | `maxresdefault.jpg` |
| Thumbs up | `7dc6efb0fe7548ae00dd6143e739f630.jpg` |venv
| Two-hand T timeout | `bc3d38ffc8a2e9a574bb54d3bffa5445.jpg` |

The included `assets/new/README.md` documents this folder. If an image is missing, MemeCV shows a readable placeholder instead of crashing.

## Detection notes

The timeout detector expects exactly two visible hands: one hand acts as a vertical stem and the other as a horizontal bar. The stem's index or middle fingertip must be near the bar hand's palm. Hand gestures have priority over face expressions when both are visible.
