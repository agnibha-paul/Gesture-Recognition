import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH      = "gesture_model.pth"
LABELS_PATH     = "label_classes.npy"
TASK_MODEL_PATH = "hand_landmarker.task"
TASK_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
CONFIDENCE_THRESHOLD = 0.8  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Download MediaPipe model if needed ───────────────────────────────────────
if not os.path.exists(TASK_MODEL_PATH):
    print("Downloading HandLandmarker model...")
    urllib.request.urlretrieve(TASK_MODEL_URL, TASK_MODEL_PATH)
    print("Download complete.")

# ─── Gesture model ────────────────────────────────────────────────────────────
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ─── Load model + labels ──────────────────────────────────────────────────────
le_classes  = np.load(LABELS_PATH, allow_pickle=True)
num_classes = len(le_classes)

model = GestureNet(input_size=126, num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded — classes: {list(le_classes)}")

# ─── MediaPipe HandLandmarker ─────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
options      = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ─── Helper: normalize landmarks ──────────────────────────────────────────────
def normalize(landmarks_21):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_21])
    origin = coords[0]
    coords = coords - origin
    scale  = np.max(np.abs(coords))
    if scale > 0:
        coords = coords / scale
    return coords.flatten().tolist()

# ─── Helper: predict gesture ──────────────────────────────────────────────────
def predict(result):
    if len(result.hand_landmarks) == 0:
        return None, 0.0

    hand_vectors = []
    for i in range(2):
        if i < len(result.hand_landmarks):
            hand_vectors.append(normalize(result.hand_landmarks[i]))
        else:
            hand_vectors.append([0.0] * 63)

    x       = torch.tensor([hand_vectors[0] + hand_vectors[1]], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits  = model(x)
        probs   = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    label      = le_classes[idx.item()]
    confidence = conf.item()
    return label, confidence

# ─── Draw landmarks ───────────────────────────────────────────────────────────
def draw_landmarks(frame, result):
    for hand_lms in result.hand_landmarks:
        h, w, _ = frame.shape
        points  = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

        # connections (MediaPipe hand topology)
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        for a, b in connections:
            cv2.line(frame, points[a], points[b], (0, 200, 0), 2)
        for pt in points:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)

# ─── Main loop ────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Running inference — press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = landmarker.detect(mp_image)

        draw_landmarks(frame, result)

        label, confidence = predict(result)

        # ── HUD ───────────────────────────────────────────────────────────────
        h, w, _ = frame.shape

        if label and confidence >= CONFIDENCE_THRESHOLD:
            # Gesture detected with high confidence
            text       = f"{label}  {confidence * 100:.1f}%"
            text_color = (0, 255, 0)
        elif label:
            # Detected but low confidence
            text       = f"? {label}  {confidence * 100:.1f}%"
            text_color = (0, 200, 255)
        else:
            text       = "No hand detected"
            text_color = (0, 0, 255)

        # Background bar for readability
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()