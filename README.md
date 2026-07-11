# Hand Gesture Recognition with MediaPipe & PyTorch

A real-time hand gesture recognition system built with MediaPipe's Hand Landmarker (Tasks API) and a PyTorch MLP classifier.

## Gestures

| Gesture | Hands |
|---|---|
| Taishakuten-In (Unlimited Void seal) | 1 |
| Perfect / OK | 1 |
| Closed Fist | 1 |
| Joined Hands | 2 |

## How it works

MediaPipe detects up to 2 hands and returns 21 landmarks per hand (x, y, z coordinates). These 126 normalized values are fed into a lightweight MLP that classifies the gesture in real time.

```
Webcam → MediaPipe HandLandmarker → 126 landmark values → GestureNet → Predicted gesture
```

## Project Structure

```
├── collect_gestures.py   # Collect training data from webcam
├── train_model.py        # Train the PyTorch classifier
├── inference.py          # Live real-time gesture recognition
├── gesture_data.csv      # Collected landmark data + labels
└── README.md
```

## Setup

**Install dependencies**
```bash
pip install mediapipe torch opencv-python scikit-learn matplotlib seaborn pandas numpy
```

> The MediaPipe `hand_landmarker.task` model file (~30MB) is downloaded automatically on first run.

## Usage

### 1. Collect gesture data
```bash
python collect_gestures.py
```
- Press `0–3` to select a gesture class
- Hold the gesture in front of your webcam — samples are captured automatically every 5 frames
- Press `Q` to quit
- Data is saved to `gesture_data.csv`

### 2. Train the model
```bash
python train_model.py
```
- Trains a 3-layer MLP on the collected landmark data
- Saves the best model to `gesture_model.pth` and labels to `label_classes.npy`
- Outputs `confusion_matrix.png` and `training_curves.png`

### 3. Run live inference
```bash
python inference.py
```
- Opens your webcam and predicts gestures in real time
- Green text = high confidence (≥80%), orange = low confidence, red = no hand detected
- Press `Q` to quit

## Model

| Property | Value |
|---|---|
| Architecture | MLP (126 → 128 → 64 → 4) |
| Parameters | ~25,000 |
| Input | 126 normalized hand landmark values |
| Output | 4 gesture classes |
| Val accuracy | 98.89% |

<img width="800" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/384a9ca7-9227-46b3-b0a2-eab353d3fb88" />


## Requirements

- Python 3.8+
- Webcam
- mediapipe >= 0.10
- torch
- opencv-python
