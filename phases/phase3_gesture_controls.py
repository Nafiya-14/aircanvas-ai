"""AirCanvas AI - Phase 3: colours, eraser, undo/redo and gesture controls.

Install:  pip install opencv-python mediapipe numpy
Run:      python phase3_gesture_controls.py

Gestures:
  Index finger up only            -> DRAW (below the toolbar)
  Index + middle fingers up       -> SELECT: hover over a toolbar button for ~0.7s to press it
  Anything else                   -> pen lifted

Toolbar (top of screen): 5 colours | ERASE | UNDO | REDO | CLEAR

Keys:  s = save PNG | z = undo | y = redo | c = clear | q = quit
"""
import os
import time
import urllib.request

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
SMOOTHING = 0.6      # 0 = no smoothing, closer to 1 = smoother but laggier
DWELL_SECONDS = 0.7  # how long to hover over a button to press it
BAR_H = 60           # toolbar height in pixels

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
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def draw_hand(frame, lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), cv2.FILLED)


def button_at(x, y, w):
    """Return the toolbar button index under (x, y), or None."""
    if y >= BAR_H or x < 0 or x >= w:
        return None
    return min(x * len(BUTTONS) // w, len(BUTTONS) - 1)


def redraw(canvas, strokes):
    """Rebuild the canvas from the stroke history (used by undo/redo)."""
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


def draw_toolbar(frame, w, state, hover_idx, hover_progress):
    bw = w // len(BUTTONS)
    for i, (kind, col, label) in enumerate(BUTTONS):
        x0, x1 = i * bw, (i + 1) * bw
        bg = col if kind == "color" else (60, 60, 60)
        cv2.rectangle(frame, (x0, 0), (x1, BAR_H), bg, cv2.FILLED)

        selected = (kind == "color" and not state["erase"] and col == state["color"]) or (
            kind == "erase" and state["erase"]
        )
        border = (255, 255, 255) if selected else (20, 20, 20)
        cv2.rectangle(frame, (x0, 0), (x1, BAR_H), border, 3 if selected else 1)

        if label:
            cv2.putText(frame, label, (x0 + 6, BAR_H // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if i == hover_idx and hover_progress > 0:
            cv2.rectangle(frame, (x0, BAR_H - 8),
                          (x0 + int(bw * min(hover_progress, 1.0)), BAR_H),
                          (0, 255, 255), cv2.FILLED)


def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    state = {"color": BUTTONS[0][1], "erase": False}
    strokes = []       # history of everything drawn (for undo/redo)
    redo_stack = []
    current = None     # stroke currently being drawn
    canvas = None
    smooth = None
    hover_idx, hover_start, hover_fired = None, 0.0, False
    start = time.time()

    def do_undo():
        if strokes:
            redo_stack.append(strokes.pop())
            redraw(canvas, strokes)

    def do_redo():
        if redo_stack:
            strokes.append(redo_stack.pop())
            redraw(canvas, strokes)

    def do_clear():
        strokes.append({"type": "clear"})
        redo_stack.clear()
        canvas[:] = 0

    def press(idx):
        kind, col, _ = BUTTONS[idx]
        if kind == "color":
            state["color"], state["erase"] = col, False
        elif kind == "erase":
            state["erase"] = True
        elif kind == "undo":
            do_undo()
        elif kind == "redo":
            do_redo()
        elif kind == "clear":
            do_clear()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            if canvas is None:
                canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((time.time() - start) * 1000))

            mode = "NO HAND"
            hover_progress = 0.0
            cursor = None

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                draw_hand(frame, lm, w, h)

                raw = np.array([lm[INDEX_TIP].x * w, lm[INDEX_TIP].y * h])
                smooth = raw if smooth is None else SMOOTHING * smooth + (1 - SMOOTHING) * raw
                pos = (int(smooth[0]), int(smooth[1]))
                cursor = pos

                index_up = finger_up(lm, INDEX_TIP, INDEX_PIP)
                middle_up = finger_up(lm, MIDDLE_TIP, MIDDLE_PIP)
                others_down = not finger_up(lm, RING_TIP, RING_PIP) and not finger_up(
                    lm, PINKY_TIP, PINKY_PIP
                )

                if index_up and middle_up:
                    # ---- SELECT mode: hover over a toolbar button ----
                    mode = "SELECT"
                    current = None
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

                elif index_up and others_down:
                    # ---- DRAW mode ----
                    hover_idx = None
                    if pos[1] < BAR_H:
                        mode = "PEN UP"  # don't draw on the toolbar
                        current = None
                    else:
                        mode = "ERASE" if state["erase"] else "DRAW"
                        col = (0, 0, 0) if state["erase"] else state["color"]
                        th = ERASER_THICKNESS if state["erase"] else PEN_THICKNESS
                        if current is None:
                            current = {"type": "line", "color": col, "thickness": th, "points": []}
                            strokes.append(current)
                            redo_stack.clear()
                        current["points"].append(pos)
                        pts = current["points"]
                        if len(pts) == 1:
                            cv2.circle(canvas, pos, th // 2, col, cv2.FILLED)
                        else:
                            cv2.line(canvas, pts[-2], pts[-1], col, th)
                else:
                    mode = "PEN UP"
                    current = None
                    hover_idx = None
            else:
                current = None
                smooth = None
                hover_idx = None

            # Overlay the canvas on the camera frame
            mask = canvas.any(axis=2)
            frame[mask] = canvas[mask]

            draw_toolbar(frame, w, state, hover_idx, hover_progress)

            if cursor is not None:
                if mode == "ERASE":
                    cv2.circle(frame, cursor, ERASER_THICKNESS // 2, (255, 255, 255), 2)
                elif mode == "DRAW":
                    cv2.circle(frame, cursor, 8, state["color"], cv2.FILLED)
                else:
                    cv2.circle(frame, cursor, 10, (0, 255, 255), 2)

            cv2.putText(frame, f"Mode: {mode}   [z] undo [y] redo [c] clear [s] save [q] quit",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("AirCanvas AI - Phase 3", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                do_clear()
                current = None
            elif key == ord("z"):
                do_undo()
                current = None
            elif key == ord("y"):
                do_redo()
            elif key == ord("s"):
                white = np.full_like(canvas, 255)
                m = canvas.any(axis=2)
                white[m] = canvas[m]
                name = f"aircanvas_{int(time.time())}.png"
                cv2.imwrite(name, white)
                print(f"Saved {name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
