import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from torch.utils.data import DataLoader

from main import (
    get_model,
    CelebADataset,
    df_test,
    images_folder,
    val_transform,
)

# -----------------------------
# 1) Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 2) DataLoader
# -----------------------------
test_loader = DataLoader(
    CelebADataset(df_test, images_folder, val_transform),
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# -----------------------------
# 3) Load trained model
# -----------------------------
model = get_model(device)
checkpoint = os.path.join(output_dir, "best_model.pth")
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

print("✅ Loaded model:", checkpoint)

# -----------------------------
# 4) Inference
# -----------------------------
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# -----------------------------
# 5) Metrics
# -----------------------------
acc = (y_pred == y_true).mean() * 100
f1 = f1_score(y_true, y_pred)

print(f"Test Accuracy: {acc:.2f}% | F1-score: {f1:.4f}")

# -----------------------------
# 6) Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Female", "Male"],
    yticklabels=["Female", "Male"],
)
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
plt.close()

# -----------------------------
# 7) ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
plt.close()

print("✅ Evaluation complete")
