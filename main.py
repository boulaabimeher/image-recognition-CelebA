import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np

# -----------------------------
# 1) Paths & Settings
# -----------------------------
main_folder = "../data/celeba/"
images_folder = os.path.join(main_folder, "img_align_celeba/img_align_celeba/")
weights_path = "inception/inception_v3_weights.pth"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
IMG_HEIGHT, IMG_WIDTH = 218, 178
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2) Load CSVs
# -----------------------------
# Attributes
df_attr = pd.read_csv(os.path.join(main_folder, "list_attr_celeba.csv"))
df_attr.set_index("image_id", inplace=True)
df_attr.replace(-1, 0, inplace=True)

# Partitions
df_partition = pd.read_csv(os.path.join(main_folder, "list_eval_partition.csv"))
df_partition.set_index("image_id", inplace=True)

# Merge for Male attribute
df = df_partition.join(df_attr["Male"])
df_train = df[df["partition"] == 0]
df_val = df[df["partition"] == 1]
df_test = df[df["partition"] == 2]

print("Train:", len(df_train), "Val:", len(df_val), "Test:", len(df_test))


# -----------------------------
# 3) Dataset Class
# -----------------------------
class CelebADataset(Dataset):
    def __init__(self, df, images_folder, transform=None):
        self.df = df
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.index[idx]
        label = int(self.df.iloc[idx]["Male"])
        img_path = os.path.join(self.images_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------------
# 4) Transforms / Augmentation
# -----------------------------
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# -----------------------------
# 5) DataLoaders
# -----------------------------
train_dataset = CelebADataset(df_train, images_folder, transform=train_transform)
val_dataset = CelebADataset(df_val, images_folder, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------
# 6) Load InceptionV3 Offline
# -----------------------------
def get_model(weights_path=None, device="cpu"):
    model = models.inception_v3(weights=None, aux_logits=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 2),  # Male/Female
    )

    if weights_path is not None and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        # Remove AuxLogits keys
        filtered_dict = {k: v for k, v in state_dict.items() if "AuxLogits" not in k}
        model.load_state_dict(filtered_dict, strict=False)
        print("✅ Loaded weights from", weights_path, "(AuxLogits ignored)")

    model = model.to(device)
    return model


model = get_model(weights_path=weights_path, device=device)

# -----------------------------
# 7) Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# -----------------------------
# 8) Training Loop with Checkpoints
# -----------------------------
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # ---- Training ----
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Print batch progress
        if batch_idx % 100 == 0 or batch_idx == len(train_loader):
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Train Loss: {running_loss / total:.4f} "
                f"Train Acc: {correct / total:.4f}",
                end="\r",
            )

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- Validation ----
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    # Print epoch summary
    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
        f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
    )

    # ---- Save best model ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        print(f"✅ Saved best model at epoch {epoch + 1} | Val Acc: {val_acc:.4f}")

# end
