"""AirCanvas AI - Phase 4: shape recognition + smart classroom (whiteboard & slides).

Install:  pip install opencv-python mediapipe numpy
Run:      python phase4_shapes_and_slides.py

SLIDES: put your slide images (PNG/JPG) in a folder called "slides" next to this
script (it is created on first run). They are shown in natural order
(slide1, slide2, ... slide10).  In PowerPoint: File > Export > PNG > All Slides.

Gestures:
  Index finger up only               -> DRAW (below the toolbar)
  Index + middle up (ring down)      -> SELECT: hover a toolbar button ~0.7s to press it
  Open palm (all 4 fingers up)       -> SWIPE left = next slide, swipe right = previous
                                        (only in SLIDES view)
  Anything else                      -> pen lifted

Toolbar: 5 colours | ERASE | UNDO | REDO | CLEAR | SHAPE | VIEW | PREV | NEXT
  SHAPE = turns rough circles / rectangles / triangles into clean shapes
  VIEW  = cycles CAMERA -> BOARD (white whiteboard) -> SLIDES

Keys: s save | z undo | y redo | c clear | r shapes on/off | v view | n/p next/prev slide | q quit
"""
import math
import os
import re
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ---- Settings you can tweak ----
PEN_THICKNESS = 6
ERASER_THICKNESS = 40
SMOOTHING = 0.6        # 0 = no smoothing, closer to 1 = smoother but laggier
DWELL_SECONDS = 0.7    # hover time to press a toolbar button
BAR_H = 60             # toolbar height in pixels
STATUS_H = 30          # space reserved for the status line at the bottom

# Shape recognition
MIN_POINTS = 12        # ignore very short strokes
MIN_SHAPE_SIZE = 40    # ignore tiny scribbles (pixels)
CLOSE_FRAC = 0.3       # stroke end must be this close to its start (fraction of size)
CIRCLE_CIRCULARITY = 0.85

# Swipe to change slides
SWIPE_WINDOW = 0.5     # seconds of hand movement considered
SWIPE_FRAC = 0.3       # distance needed, as a fraction of screen width
SWIPE_COOLDOWN = 1.0   # seconds between swipes

# Toolbar buttons: (kind, colour in BGR or None, label)
BUTTONS = [
    ("color", (255, 0, 0), ""),      # blue
    ("color", (0, 0, 255), ""),      # red
    ("color", (0, 200, 0), ""),      # green
    ("color", (0, 165, 255), ""),    # orange
    ("color", (200, 0, 200), ""),    # purple
    ("erase", None, "ERASE"),
    ("undo", None, "UNDO"),
    ("redo", None, "REDO"),
    ("clear", None, "CLEAR"),
    ("shape", None, "SHAPE"),
    ("view", None, "VIEW"),
    ("prev", None, "PREV"),
    ("next", None, "NEXT"),
]


# ---------------------------------------------------------------- helpers
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def draw_hand(img, lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(img, p, 3, (0, 0, 255), cv2.FILLED)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def fit_to_area(img, w, h):
    """Fit a slide image into the area between the toolbar and the status line."""
    area_h = h - BAR_H - STATUS_H
    ih, iw = img.shape[:2]
    scale = min(w / iw, area_h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 255, np.uint8)
    x0 = (w - nw) // 2
    y0 = BAR_H + (area_h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def load_slides(w, h):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides")
    os.makedirs(folder, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted((f for f in os.listdir(folder) if f.lower().endswith(exts)), key=natural_key)
    slides = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is not None:
            slides.append(fit_to_area(img, w, h))
    return slides


# ---------------------------------------------------------- shape recognition
def recognize_shape(points):
    """Return a list of points for a clean shape, or None to keep the freehand stroke."""
    if len(points) < MIN_POINTS:
        return None

    pts = np.array(points, dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    size = max(bw, bh)
    if size < MIN_SHAPE_SIZE:
        return None

    # Must be a (roughly) closed stroke
    gap = np.linalg.norm(pts[0].astype(float) - pts[-1].astype(float))
    if gap > CLOSE_FRAC * size:
        return None

    hull = cv2.convexHull(pts)
    area = cv2.contourArea(hull)
    peri = cv2.arcLength(hull, True)
    if area <= 0 or peri <= 0 or area < 0.2 * bw * bh:
        return None  # basically a line or a thin scribble

    circularity = 4 * math.pi * area / (peri * peri)
    approx = cv2.approxPolyDP(hull, 0.05 * peri, True).reshape(-1, 2)

    if circularity > CIRCLE_CIRCULARITY:
        m = cv2.moments(hull)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        r = math.sqrt(area / math.pi)
        return [
            (int(cx + r * math.cos(t)), int(cy + r * math.sin(t)))
            for t in np.linspace(0, 2 * math.pi, 73)
        ]

    if len(approx) == 3:
        tri = [tuple(int(v) for v in p) for p in approx]
        return tri + [tri[0]]

    if len(approx) == 4:
        rect = cv2.minAreaRect(hull)
        angle = abs(rect[2]) % 90
        if angle < 12 or angle > 78:  # nearly upright -> snap to axis-aligned
            hx, hy, hw, hh = cv2.boundingRect(hull)
            box = [(hx, hy), (hx + hw, hy), (hx + hw, hy + hh), (hx, hy + hh)]
        else:
            box = [tuple(int(v) for v in p) for p in cv2.boxPoints(rect)]
        return box + [box[0]]

    return None


# ---------------------------------------------------------------- drawing layer
def redraw(canvas, strokes):
    """Rebuild the canvas from the stroke history (used by undo/redo/shapes)."""
    canvas[:] = 0
    for s in strokes:
        if s["type"] == "clear":
            canvas[:] = 0
            continue
        pts, col, th = s["points"], s["color"], s["thickness"]
        if len(pts) == 1:
            cv2.circle(canvas, pts[0], th // 2, col, cv2.FILLED)
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], col, th)


class Layer:
    """One drawing surface (the whiteboard, or one slide) with its own undo history."""

    def __init__(self, shape):
        self.canvas = np.zeros(shape, np.uint8)
        self.strokes = []
        self.redo_stack = []
        self.current = None

    def add_point(self, pos, color, thickness, erase):
        if self.current is None:
            self.current = {"type": "line", "color": color, "thickness": thickness,
                            "points": [], "erase": erase}
            self.strokes.append(self.current)
            self.redo_stack.clear()
        pts = self.current["points"]
        pts.append(pos)
        if len(pts) == 1:
            cv2.circle(self.canvas, pos, thickness // 2, color, cv2.FILLED)
        else:
            cv2.line(self.canvas, pts[-2], pts[-1], color, thickness)

    def end_stroke(self, shape_on):
        s, self.current = self.current, None
        if s is None or not shape_on or s["erase"]:
            return
        clean = recognize_shape(s["points"])
        if clean is not None:
            s["points"] = clean
            redraw(self.canvas, self.strokes)

    def undo(self):
        self.current = None
        if self.strokes:
            self.redo_stack.append(self.strokes.pop())
            redraw(self.canvas, self.strokes)

    def redo(self):
        if self.redo_stack:
            self.strokes.append(self.redo_stack.pop())
            redraw(self.canvas, self.strokes)

    def clear(self):
        self.current = None
        self.strokes.append({"type": "clear"})
        self.redo_stack.clear()
        self.canvas[:] = 0


# ---------------------------------------------------------------- UI
def button_at(x, y, w):
    if y >= BAR_H or x < 0 or x >= w:
        return None
    return min(x * len(BUTTONS) // w, len(BUTTONS) - 1)


def draw_toolbar(img, w, S, hover_idx, hover_progress, view_label):
    bw = w // len(BUTTONS)
    fs = 0.5 if bw >= 90 else 0.38
    for i, (kind, col, label) in enumerate(BUTTONS):
        x0, x1 = i * bw, (i + 1) * bw
        bg = col if kind == "color" else (60, 60, 60)
        cv2.rectangle(img, (x0, 0), (x1, BAR_H), bg, cv2.FILLED)

        selected = (
            (kind == "color" and not S["erase"] and col == S["color"])
            or (kind == "erase" and S["erase"])
            or (kind == "shape" and S["shape_on"])
        )
        cv2.rectangle(img, (x0, 0), (x1, BAR_H),
                      (255, 255, 255) if selected else (20, 20, 20), 3 if selected else 1)

        if kind == "view":
            label = view_label
        if label:
            cv2.putText(img, label, (x0 + 6, BAR_H // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)

        if i == hover_idx and hover_progress > 0:
            cv2.rectangle(img, (x0, BAR_H - 8),
                          (x0 + int(bw * min(hover_progress, 1.0)), BAR_H),
                          (0, 255, 255), cv2.FILLED)


# ---------------------------------------------------------------- main
def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    S = {"color": BUTTONS[0][1], "erase": False, "shape_on": True,
         "view_idx": 0, "slide_idx": 0, "toast": "", "toast_until": 0.0}
    views = ["camera", "board"]
    slides = []
    layers = None

    def cur_view():
        return views[S["view_idx"]]

    def cur_layer():
        if cur_view() == "slides":
            return layers["slides"][S["slide_idx"]] # type: ignore
        return layers["board"] # type: ignore

    def toast(msg):
        S["toast"], S["toast_until"] = msg, time.time() + 1.5

    def cycle_view():
        cur_layer().end_stroke(False)
        S["view_idx"] = (S["view_idx"] + 1) % len(views)
        toast(f"View: {cur_view().upper()}")

    def change_slide(delta):
        if cur_view() != "slides":
            toast("Switch VIEW to SLIDES first" if slides else "No slides found in 'slides' folder")
            return
        cur_layer().end_stroke(False)
        S["slide_idx"] = max(0, min(len(slides) - 1, S["slide_idx"] + delta))
        toast(f"Slide {S['slide_idx'] + 1} / {len(slides)}")

    def press(idx):
        kind, col, _ = BUTTONS[idx]
        if kind == "color":
            S["color"], S["erase"] = col, False
        elif kind == "erase":
            S["erase"] = True
        elif kind == "undo":
            cur_layer().undo()
        elif kind == "redo":
            cur_layer().redo()
        elif kind == "clear":
            cur_layer().clear()
        elif kind == "shape":
            S["shape_on"] = not S["shape_on"]
            toast(f"Shape recognition {'ON' if S['shape_on'] else 'OFF'}")
        elif kind == "view":
            cycle_view()
        elif kind == "prev":
            change_slide(-1)
        elif kind == "next":
            change_slide(+1)

    smooth = None
    hover_idx, hover_start, hover_fired = None, 0.0, False
    swipe_hist = deque()
    last_swipe = 0.0
    start = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, cam = cap.read()
            if not ok:
                break
            cam = cv2.flip(cam, 1)
            h, w, _ = cam.shape

            if layers is None:  # first frame: we now know the frame size
                slides = load_slides(w, h)
                layers = {"board": Layer(cam.shape), "slides": [Layer(cam.shape) for _ in slides]}
                if slides:
                    views = ["camera", "board", "slides"]
                    print(f"Loaded {len(slides)} slide(s).")
                else:
                    print("No slides found. Add PNG/JPG images to the 'slides' folder to use SLIDES view.")

            rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((time.time() - start) * 1000))

            cam_vis = cam.copy()
            mode = "NO HAND"
            hover_progress = 0.0
            cursor = None
            layer = cur_layer()

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                draw_hand(cam_vis, lm, w, h)

                raw = np.array([lm[INDEX_TIP].x * w, lm[INDEX_TIP].y * h])
                smooth = raw if smooth is None else SMOOTHING * smooth + (1 - SMOOTHING) * raw
                pos = (int(smooth[0]), int(smooth[1]))
                cursor = pos

                index_up = finger_up(lm, INDEX_TIP, INDEX_PIP)
                middle_up = finger_up(lm, MIDDLE_TIP, MIDDLE_PIP)
                ring_up = finger_up(lm, RING_TIP, RING_PIP)
                pinky_up = finger_up(lm, PINKY_TIP, PINKY_PIP)

                palm = index_up and middle_up and ring_up and pinky_up
                select = index_up and middle_up and not ring_up
                draw = index_up and not middle_up and not ring_up and not pinky_up

                if palm:
                    # ---- PALM: swipe left/right to change slides ----
                    mode = "PALM"
                    layer.end_stroke(S["shape_on"])
                    hover_idx = None
                    now = time.time()
                    swipe_hist.append((now, lm[0].x * w))
                    while swipe_hist and now - swipe_hist[0][0] > SWIPE_WINDOW:
                        swipe_hist.popleft()
                    if (cur_view() == "slides" and now - last_swipe > SWIPE_COOLDOWN
                            and len(swipe_hist) >= 3):
                        dx = swipe_hist[-1][1] - swipe_hist[0][1]
                        if abs(dx) > SWIPE_FRAC * w:
                            change_slide(+1 if dx < 0 else -1)
                            last_swipe = now
                            swipe_hist.clear()
                else:
                    swipe_hist.clear()

                    if select:
                        # ---- SELECT: hover over a toolbar button ----
                        mode = "SELECT"
                        layer.end_stroke(S["shape_on"])
                        idx = button_at(pos[0], pos[1], w)
                        if idx is None:
                            hover_idx = None
                        else:
                            if idx != hover_idx:
                                hover_idx, hover_start, hover_fired = idx, time.time(), False
                            elapsed = time.time() - hover_start
                            hover_progress = elapsed / DWELL_SECONDS
                            if not hover_fired and elapsed >= DWELL_SECONDS:
                                press(idx)
                                hover_fired = True

                    elif draw and pos[1] >= BAR_H:
                        # ---- DRAW / ERASE ----
                        hover_idx = None
                        erase = S["erase"]
                        mode = "ERASE" if erase else "DRAW"
                        col = (0, 0, 0) if erase else S["color"]
                        th = ERASER_THICKNESS if erase else PEN_THICKNESS
                        layer.add_point(pos, col, th, erase)

                    else:
                        mode = "PEN UP"
                        hover_idx = None
                        layer.end_stroke(S["shape_on"])
            else:
                layer.end_stroke(S["shape_on"])
                smooth = None
                hover_idx = None
                swipe_hist.clear()

            # ---- compose the picture (view may have changed this frame) ----
            view = cur_view()
            layer = cur_layer()

            if view == "camera":
                out = cam_vis
                base = np.full_like(cam, 255)
            elif view == "board":
                out = np.full_like(cam, 255)
                base = np.full_like(cam, 255)
            else:
                out = slides[S["slide_idx"]].copy()
                base = slides[S["slide_idx"]]

            mask = layer.canvas.any(axis=2)
            out[mask] = layer.canvas[mask]

            if view != "camera":  # small camera preview so you can see your hand
                iw, ih = w // 5, h // 5
                x0, y0 = w - iw - 10, h - ih - STATUS_H - 4
                out[y0:y0 + ih, x0:x0 + iw] = cv2.resize(cam_vis, (iw, ih))
                cv2.rectangle(out, (x0, y0), (x0 + iw, y0 + ih), (80, 80, 80), 2)

            draw_toolbar(out, w, S, hover_idx, hover_progress, view.upper())

            if cursor is not None:
                if mode == "ERASE":
                    cv2.circle(out, cursor, ERASER_THICKNESS // 2, (120, 120, 120), 2)
                elif mode == "DRAW":
                    cv2.circle(out, cursor, 8, S["color"], cv2.FILLED)
                else:
                    cv2.circle(out, cursor, 10, (0, 200, 255), 2)

            if time.time() < S["toast_until"]:
                cv2.putText(out, S["toast"], (20, BAR_H + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                cv2.putText(out, S["toast"], (20, BAR_H + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            slide_info = f" {S['slide_idx'] + 1}/{len(slides)}" if view == "slides" else ""
            status = (f"Mode: {mode} | View: {view}{slide_info} | "
                      "[z]undo [y]redo [c]clear [r]shapes [v]view [n/p]slide [s]save [q]quit")
            cv2.rectangle(out, (0, h - STATUS_H), (w, h), (30, 30, 30), cv2.FILLED)
            cv2.putText(out, status, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("AirCanvas AI - Phase 4", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                layer.clear()
            elif key == ord("z"):
                layer.undo()
            elif key == ord("y"):
                layer.redo()
            elif key == ord("r"):
                press(next(i for i, b in enumerate(BUTTONS) if b[0] == "shape"))
            elif key == ord("v"):
                cycle_view()
            elif key == ord("n"):
                change_slide(+1)
            elif key == ord("p"):
                change_slide(-1)
            elif key == ord("s"):
                saved = base.copy()
                m = layer.canvas.any(axis=2)
                saved[m] = layer.canvas[m]
                name = f"aircanvas_{int(time.time())}.png"
                cv2.imwrite(name, saved)
                print(f"Saved {name}")
                toast(f"Saved {name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
