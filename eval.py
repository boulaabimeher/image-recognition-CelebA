import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import numpy as np

from img_recov1 import CelebADataset, get_model  # <-- adapt to your model file


# -----------------------------------------------------------
# SAVE PLOTS WITHOUT SHOWING THEM
# -----------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_roc_curve(y_true, y_score, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------
# MAIN EVAL FUNCTION
# -----------------------------------------------------------
def evaluate(model_path, data_root, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Load Model
    # -----------------------------
    model = get_model(weights_path=model_path, device=device)  # your model class
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # -----------------------------
    # Load Dataset
    # -----------------------------
    dataset = CelebADataset(data_root, split="test")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    y_true = []
    y_pred = []
    y_scores = []

    # -----------------------------
    # Evaluation Loop
    # -----------------------------
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()

            preds = (probs > 0.5).int()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    # Convert lists to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # -----------------------------
    # Print Metrics to File
    # -----------------------------
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred))

    print("✔ classification_report.txt saved")

    # -----------------------------
    # Save Figures
    # -----------------------------
    save_confusion_matrix(
        y_true, y_pred, os.path.join(output_dir, "confusion_matrix.png")
    )

    save_roc_curve(y_true, y_scores, os.path.join(output_dir, "roc_curve.png"))

    print("✔ confusion_matrix.png saved")
    print("✔ roc_curve.png saved")
    print("Evaluation complete!")


# -----------------------------------------------------------
# Run eval.py directly
# -----------------------------------------------------------
if __name__ == "__main__":
    evaluate(
        model_path="model/best_model.pth", data_root="data/celeba", output_dir="output"
    )
