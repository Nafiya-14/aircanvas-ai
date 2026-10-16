# AirCanvas AI ✋🎨

**A gesture-controlled virtual whiteboard for teachers.** Write, draw and run a slide presentation with hand movements in front of a webcam — no keyboard, mouse or touchscreen needed.

> 📽️ **Demo:** _add a GIF or video link here (record your screen while using the app)_

## Features

- **Real-time hand tracking** with MediaPipe — works with a normal webcam
- **Air writing:** draw with your index finger, with smoothing for clean lines
- **Smart shape recognition:** rough circles, rectangles and triangles snap into clean shapes
- **Gesture toolbar:** pick colours, erase, undo, redo and clear by hovering your fingers over buttons
- **Smart classroom views:** switch between your camera, a white whiteboard and your presentation slides
- **Hands-free slides:** swipe your open palm left or right to change slide; annotations are kept per slide
- **Save your work** as PNG images

## How it works

```
Webcam  →  MediaPipe Hand Landmarker  →  Gesture logic  →  OpenCV canvas  →  Screen
(frames)   (21 landmarks per hand)       (draw / select /    (strokes, undo,    (whiteboard,
                                          palm swipe)         shape snapping)    slides, camera)
```

1. Each webcam frame is passed to MediaPipe's hand landmark model.
2. The script checks which fingers are raised to decide the current gesture.
3. Index fingertip positions are smoothed and recorded as strokes on a transparent canvas layer.
4. When a stroke ends, its convex hull is analysed (circularity, polygon approximation) to detect circles, rectangles and triangles and replace the freehand stroke with a clean shape.
5. Strokes are stored as a history, which makes undo/redo and per-slide annotations straightforward.

## Gestures

| Gesture | Action |
| --- | --- |
| ☝️ Index finger only | **Draw** (below the toolbar) |
| ✌️ Index + middle fingers (ring finger down) | **Select** — hover a toolbar button for ~0.7 s to press it |
| 🖐️ Open palm, swipe left / right | **Next / previous slide** (SLIDES view only) |
| Anything else | Pen lifted |

**Toolbar:** 5 colours · ERASE · UNDO · REDO · CLEAR · SHAPE (shape recognition on/off) · VIEW (camera → board → slides) · PREV · NEXT

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `s` | Save the current board/slide as a PNG |
| `z` / `y` | Undo / redo |
| `c` | Clear |
| `r` | Toggle shape recognition |
| `v` | Cycle view |
| `n` / `p` | Next / previous slide |
| `q` | Quit |

## Getting started

**Requirements:** Python 3.10+ and a webcam.

```bash
git clone https://github.com/<your-username>/aircanvas-ai.git
cd aircanvas-ai

# create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python aircanvas.py
```

On the first run the app downloads the MediaPipe hand landmark model (~8 MB) automatically, so you need an internet connection once.

### Using your own slides

Export your presentation as images and put them in the `slides/` folder, named so they sort in order (`slide1.png`, `slide2.png`, …). In PowerPoint: **File → Export → PNG → All Slides**. Restart the app, then press the VIEW button (or `v`) until you reach SLIDES.

## Configuration

Tweak the constants at the top of `aircanvas.py`:

| Setting | What it does |
| --- | --- |
| `SMOOTHING` | Higher = smoother lines but more lag |
| `DWELL_SECONDS` | How long to hover to press a toolbar button |
| `PEN_THICKNESS`, `ERASER_THICKNESS` | Brush sizes |
| `CIRCLE_CIRCULARITY`, `CLOSE_FRAC` | How eagerly shapes snap |
| `SWIPE_FRAC`, `SWIPE_COOLDOWN` | Swipe sensitivity for slide changes |

## Tips

- Use good, even lighting and keep your palm facing the camera.
- Keep your ring and little fingers curled while drawing.
- Sit about an arm's length from the webcam so your whole hand is visible.

## Troubleshooting

- **`module 'mediapipe' has no attribute 'solutions'`** — this project uses MediaPipe's newer Tasks API, so make sure you run `aircanvas.py` rather than older scripts that use `mp.solutions`.
- **`numpy` import errors** — recreate the virtual environment and reinstall: `pip install --no-cache-dir -r requirements.txt`.
- **Camera doesn't open** — close other apps using the webcam, or change `cv2.VideoCapture(0)` to `1`.

## Tech stack

Python · OpenCV · MediaPipe · NumPy

## Roadmap

- [ ] Browser version (React + Canvas API + MediaPipe Tasks for the web)
- [ ] Ellipse / line / arrow recognition
- [ ] More pen sizes and a colour picker
- [ ] Load PDF slides directly
- [ ] Two-hand gestures (zoom, move)

## License

Add a license of your choice (MIT is a common default for portfolio projects).
