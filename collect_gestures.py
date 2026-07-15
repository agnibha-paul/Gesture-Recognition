import cv2
import csv
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark as landmark_module
import urllib.request

# ------ Config --------
GESTURES        = ["Peace", "Perfect", "Joined hands", "Closed fist", "Rabbit", "Call me", "Thumbs Up", 'Thumbs Down']
OUTPUT_CSV      = "gesture_data.csv"
CAPTURE_EVERY_N = 5          # capture a sample every N frames when hand detected
SAMPLES_TARGET  = 300        # print a warning when you hit this per gesture
MODEL_PATH      = "hand_landmarker.task"
MODEL_URL       = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# ------- Download model if needed --------
if not os.path.exists(MODEL_PATH):
    print(f"Downloading HandLandmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# -------- Build HandLandmarker ------------
base_options    = python.BaseOptions(model_asset_path=MODEL_PATH)
options         = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker      = vision.HandLandmarker.create_from_options(options)

# ------------ CSV setup -----------
file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
csv_file    = open(OUTPUT_CSV, "a", newline="")
csv_writer  = csv.writer(csv_file)

if not file_exists:
    # header: 126 landmark values (2 hands * 21 landmarks * xyz) + label
    lm_headers = [f"lm{i}_{axis}" for i in range(42) for axis in ("x", "y", "z")]
    csv_writer.writerow(lm_headers + ["label"])

# ------------ Helper: normalize landmarks --------------
def normalize(landmarks_21):
    """
    Normalize 21 landmarks relative to wrist (landmark 0).
    Returns flat list of 63 values.
    """
    coords  = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_21])
    origin  = coords[0]                  # wrist
    coords  = coords - origin            # translate to origin
    scale   = np.max(np.abs(coords))     # scale by max extent
    if scale > 0:
        coords = coords / scale
    return coords.flatten().tolist()

# ---------- Count existing samples per gesture ------------
def count_samples():
    counts = {g: 0 for g in GESTURES}
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row and row[-1] in counts:
                    counts[row[-1]] += 1
    return counts

# ------------ Main -----------
def main():
    print("\n=== Gesture Data Collector ===")
    print("Gestures available:")
    for i, g in enumerate(GESTURES):
        print(f"  [{i}] {g}")
    print("\nEnter the gesture number to collect, then press Q to switch/quit.\n")

    current_gesture_idx = None
    frame_count         = 0
    sample_count        = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result      = landmarker.detect(mp_image)

        hand_detected = len(result.hand_landmarks) > 0

        # Auto-capture every N frames if gesture selected and hand detected
        if current_gesture_idx is not None and hand_detected and frame_count % CAPTURE_EVERY_N == 0:
            gesture_name = GESTURES[current_gesture_idx]

            # Build 126-value vector (2 hands, zero-pad missing hand)
            hand_vectors = []
            for i in range(2):
                if i < len(result.hand_landmarks):
                    hand_vectors.append(normalize(result.hand_landmarks[i]))
                else:
                    hand_vectors.append([0.0] * 63)  # zero-pad

            row = hand_vectors[0] + hand_vectors[1] + [gesture_name]
            csv_writer.writerow(row)
            csv_file.flush()
            sample_count += 1

        # ----------- Draw landmarks --------
        annotated = frame.copy()
        for hand_lms in result.hand_landmarks:
            for lm in hand_lms:
                h, w, _ = frame.shape
                cx, cy  = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        # ------------------ HUD ---------------------
        gesture_label = GESTURES[current_gesture_idx] if current_gesture_idx is not None else f"None (press 0-{len(GESTURES)-1})"
        status_color  = (0, 255, 0) if hand_detected else (0, 0, 255)

        cv2.putText(annotated, f"Gesture : {gesture_label}",  (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Captured: {sample_count}",   (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Hand    : {'YES' if hand_detected else 'NO'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(annotated, f"Keys: 0-{len(GESTURES)-1} select gesture | Q quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Gesture Collector", annotated)

        key = cv2.waitKey(1) & 0xFF
        print(key)
        if key == ord('q') or key == ord('Q'):
            break
        elif ord('0') <= key <= ord('7'):
            idx = key - ord('0')
            if idx < len(GESTURES):
                current_gesture_idx = idx
                sample_count        = 0
                counts              = count_samples()
                print(f"\nSwitched to: {GESTURES[idx]} (already have {counts[GESTURES[idx]]} samples)")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    landmarker.close()

    print("\n=== Collection complete ===")
    counts = count_samples()
    for g, c in counts.items():
        print(f"  {g}: {c} samples")

if __name__ == "__main__":
    main()
