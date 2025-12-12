import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np

# -----------------------------
# 1) Paths & Device
# -----------------------------
main_folder = "data/celeba/"
images_folder = os.path.join(main_folder, "img_align_celeba/img_align_celeba/")
weights_path = "inception/inception_v3_weights.pth"  # offline weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("output", exist_ok=True)

# -----------------------------
# 2) Load Attributes & Partitions
# -----------------------------
df_attr = pd.read_csv(os.path.join(main_folder, "list_attr_celeba.csv"))
df_attr.set_index("image_id", inplace=True)
df_attr.replace(-1, 0, inplace=True)

df_partition = pd.read_csv(os.path.join(main_folder, "list_eval_partition.csv"))
df_partition.set_index("image_id", inplace=True)

# Merge Male attribute with partition
df = df_partition.join(df_attr["Male"])
df_train = df[df["partition"] == 0]
df_val = df[df["partition"] == 1]
df_test = df[df["partition"] == 2]

print("Train:", len(df_train))
print("Val:  ", len(df_val))
print("Test: ", len(df_test))


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
        label = int(self.df.iloc[idx, 0])  # Male: 0/1
        img_path = os.path.join(self.images_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# -----------------------------
# 4) Data Augmentation
# -----------------------------
IMG_HEIGHT, IMG_WIDTH = 218, 178
BATCH_SIZE = 16

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ]
)

train_dataset = CelebADataset(df_train, images_folder, transform=train_transform)
val_dataset = CelebADataset(df_val, images_folder, transform=val_transform)
test_dataset = CelebADataset(df_test, images_folder, transform=val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)


# -----------------------------
# 5) Load InceptionV3 Offline
# -----------------------------
def get_model(weights_path=None, device="cpu"):
    import torch
    import torch.nn as nn
    from torchvision import models
    import os

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
        # Remove aux logits keys
        filtered_dict = {k: v for k, v in state_dict.items() if "AuxLogits" not in k}
        model.load_state_dict(filtered_dict, strict=False)
        print("✅ Loaded weights from", weights_path, "(AuxLogits ignored)")

    model = model.to(device)
    return model


# Initialize the model with your local weights
model = get_model(weights_path=weights_path, device=device)  # your local weights


# -----------------------------
# 6) Loss, Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# -----------------------------
# 7) Training Loop with Checkpoints
# -----------------------------
NUM_EPOCHS = 1
best_val_loss = float("inf")

train_losses, val_losses = [], []
train_acc, val_acc = [], []

for epoch in range(NUM_EPOCHS):
    # ----- Training -----
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
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

    train_loss = running_loss / total
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_acc.append(train_accuracy)

    # ----- Validation -----
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_acc.append(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f}"
    )

    # ----- Checkpoint -----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            "checkpoints/best_model.pth",
        )
        print(f"✅ Saved checkpoint at epoch {epoch + 1}")

# -----------------------------
# 8) Test Evaluation
# -----------------------------
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = correct / total
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")
