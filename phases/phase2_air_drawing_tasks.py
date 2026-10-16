"""AirCanvas AI - Phase 2: smooth air drawing (MediaPipe Tasks API version).

Works with current mediapipe releases (no mp.solutions needed).

Install:  pip install opencv-python mediapipe numpy
Run:      python phase2_air_drawing_tasks.py
          (downloads hand_landmarker.task once on first run, ~8 MB)

Gestures:
  Index finger up, other fingers down  -> DRAW
  Index + middle fingers up            -> pen lifted (move without drawing)

Keys:  c = clear canvas | s = save drawing as PNG | q = quit
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

# Landmark ids: (tip, pip joint) for each finger
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

# Pairs of landmark ids to connect when drawing the hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm edge
]

PEN_COLOR = (255, 0, 0)  # BGR (blue). Colours come in Phase 3.
PEN_THICKNESS = 6
SMOOTHING = 0.6  # 0 = no smoothing, closer to 1 = smoother but laggier


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def finger_up(lm, tip, pip):
    """A finger counts as 'up' if its tip is above its middle joint (smaller y)."""
    return lm[tip].y < lm[pip].y


def draw_hand(frame, lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), cv2.FILLED)


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

    canvas = None
    prev = None    # previous smoothed fingertip position
    smooth = None  # current smoothed fingertip position
    start = time.time()

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
            timestamp_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            mode = "NO HAND"

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                draw_hand(frame, lm, w, h)

                raw = np.array([lm[INDEX_TIP].x * w, lm[INDEX_TIP].y * h])
                smooth = raw if smooth is None else SMOOTHING * smooth + (1 - SMOOTHING) * raw
                pos = (int(smooth[0]), int(smooth[1]))

                index_up = finger_up(lm, INDEX_TIP, INDEX_PIP)
                middle_up = finger_up(lm, MIDDLE_TIP, MIDDLE_PIP)
                others_down = not finger_up(lm, RING_TIP, RING_PIP) and not finger_up(
                    lm, PINKY_TIP, PINKY_PIP
                )

                if index_up and not middle_up and others_down:
                    mode = "DRAW"
                    if prev is not None:
                        cv2.line(canvas, prev, pos, PEN_COLOR, PEN_THICKNESS)
                    prev = pos
                    cv2.circle(frame, pos, 10, PEN_COLOR, cv2.FILLED)
                else:
                    mode = "PEN UP"
                    prev = None
                    cv2.circle(frame, pos, 10, (0, 255, 255), 2)
            else:
                prev = None
                smooth = None

            # Overlay the canvas on the camera frame
            mask = canvas.any(axis=2)
            frame[mask] = canvas[mask]

            cv2.putText(frame, f"Mode: {mode}   [c] clear  [s] save  [q] quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("AirCanvas AI - Phase 2", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                canvas[:] = 0
                prev = None
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
