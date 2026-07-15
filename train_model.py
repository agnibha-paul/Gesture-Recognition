import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Config ----------
CSV_PATH    = "gesture_data.csv"
MODEL_PATH  = "gesture_model.pth"
EPOCHS      = 50
BATCH_SIZE  = 32
TEST_SIZE   = 0.2
RANDOM_SEED = 42
LR          = 1e-3
PATIENCE    = 10          # early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")

# ------------------- Dataset ----------------
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------ Model --------------------
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

# -------------------- Load data -------------------------
print("Loading data...")
df = pd.read_csv(CSV_PATH)

print(f"Total samples : {len(df)}")
print(f"Class distribution:\n{df['label'].value_counts()}\n")

X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values

le     = LabelEncoder()
y_enc  = le.fit_transform(y)

print(f"Classes : {list(le.classes_)}")
print(f"Input shape : {X.shape}\n")

# ------------- Split ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_enc
)
print(f"Train samples : {len(X_train)}")
print(f"Val samples   : {len(X_val)}\n")

train_loader = DataLoader(GestureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(GestureDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

# ------------------ Init model ----------------------
model     = GestureNet(input_size=126, num_classes=len(le.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

print(model)
print(f"\nTotal params: {sum(p.numel() for p in model.parameters()):,}\n")

# ----------------------------- Training loop ---------------------------
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

best_val_acc   = 0.0
patience_count = 0

print("Training...\n")
for epoch in range(1, EPOCHS + 1):

    # ------ Train ---
    model.train()
    train_loss, train_correct = 0.0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * len(X_batch)
        train_correct += (logits.argmax(1) == y_batch).sum().item()

    train_loss /= len(X_train)
    train_acc   = train_correct / len(X_train)

    # ----------- Validate ----------
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits    = model(X_batch)
            loss      = criterion(logits, y_batch)
            val_loss += loss.item() * len(X_batch)
            val_correct += (logits.argmax(1) == y_batch).sum().item()

    val_loss /= len(X_val)
    val_acc   = val_correct / len(X_val)

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

    # --------- Early stopping -----------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best val acc: {best_val_acc:.4f})")
            break

# ------------ Evaluate best model ------------
print("\n=== Evaluation (best model) ===")
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(DEVICE)
        preds   = model(X_batch).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())

y_pred_cls = le.inverse_transform(all_preds)
y_true_cls = le.inverse_transform(all_true)

print(f"\nBest val accuracy: {best_val_acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_true_cls, y_pred_cls))

# ------------------ Confusion matrix ---------------------
cm = confusion_matrix(y_true_cls, y_pred_cls, labels=le.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Saved: confusion_matrix.png")

# -------------------- Training curves --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_acc"], label="Train")
ax1.plot(history["val_acc"],   label="Val")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()

ax2.plot(history["train_loss"], label="Train")
ax2.plot(history["val_loss"],   label="Val")
ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
print("Saved: training_curves.png")

# --------------- Save label classes -----------------------
np.save("label_classes.npy", le.classes_)
print(f"Saved: {MODEL_PATH}")
print(f"Saved: label_classes.npy")
