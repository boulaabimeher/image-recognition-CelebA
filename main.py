import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# -----------------------------
# 1) Paths & Settings
# -----------------------------
main_folder = "../data/celeba/"
images_folder = os.path.join(main_folder, "img_align_celeba/img_align_celeba/")
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
IMG_HEIGHT, IMG_WIDTH = 218, 178
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2) Load CSVs
# -----------------------------
df_attr = pd.read_csv(os.path.join(main_folder, "list_attr_celeba.csv"))
df_attr.set_index("image_id", inplace=True)
df_attr.replace(-1, 0, inplace=True)

df_partition = pd.read_csv(os.path.join(main_folder, "list_eval_partition.csv"))
df_partition.set_index("image_id", inplace=True)

df = df_partition.join(df_attr["Male"])
df_train = df[df["partition"] == 0]
df_val = df[df["partition"] == 1]
df_test = df[df["partition"] == 2]


# -----------------------------
# 3) Dataset
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
# 4) Transforms
# -----------------------------
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# -----------------------------
# 5) Model
# -----------------------------
def get_model(device="cpu"):
    model = models.inception_v3(
        weights=None,
        aux_logits=False,
        init_weights=True,
    )

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 2),
    )

    return model.to(device)


# =============================
# TRAINING (ONLY WHEN RUN)
# =============================
if __name__ == "__main__":
    train_loader = DataLoader(
        CelebADataset(df_train, images_folder, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        CelebADataset(df_val, images_folder, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = get_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=running_loss / total,
                acc=correct / total,
            )

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"\nVal Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model.pth"),
            )
            print("✅ Saved best model")
