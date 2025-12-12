import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from torch.utils.data import DataLoader

# Import your shared code from main.py
from main import get_model, CelebADataset, df_test, images_folder

# -----------------------------
# 1) Device and output folder
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 2) Prepare test DataLoader
# -----------------------------
# Use the dataset class defined in main.py
test_dataset = CelebADataset(df_test, images_folder)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -----------------------------
# 3) Load InceptionV3 model offline
# -----------------------------
weights_path = "inception/inception_v3_weights.pth"
model = get_model(weights_path=weights_path, device=device)
model.eval()

# -----------------------------
# 4) Run evaluation
# -----------------------------
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of Male=1
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# -----------------------------
# 5) Metrics
# -----------------------------
acc = 100 * np.mean(all_preds == all_labels)
f1 = f1_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.2f}% | F1-score: {f1:.4f}")

# -----------------------------
# 6) Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(
    os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
)
plt.close()

# -----------------------------
# 7) ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

print("✅ Evaluation complete. Plots saved to", output_dir)
