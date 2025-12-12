# 🧠 Image Recognition on CelebA with PyTorch  
## 🔍 Gender Classification Using InceptionV3 (Offline Weights)

This project implements a complete **PyTorch image recognition pipeline** using the **CelebA** dataset to classify **gender (Male/Female)**.  
It is fully compatible with **offline environments** such as HPC clusters (no internet), thanks to local `.pth` weight loading.

---

## 🚀 Features

- **Offline Deep Learning** (no pretrained download required)
- **Custom InceptionV3 architecture**
- **Full training loop with tqdm progress bars**
- **Automatic best-model checkpointing**
- **Evaluation script with accuracy, F1-score, and confusion matrix**
- **Clean modular structure identical for train/eval**

---


## Run training:

bash: 

python main.py \
  --data_dir /path/to/celeba \
  --weights inception/inception_v3_weights.pth \
  --epochs 10 \
  --batch_size 64

---

## 🧪 Evaluation

Run evaluation:

python eval.py \
  --data_dir /path/to/celeba \
  --weights outputs/best_model.pth


## Outputs include:

Accuracy

F1-score

Classification report

Confusion matrix plot

📊 Confusion Matrix Example

The evaluation script will display a confusion matrix similar to:

[[ TN  FP ]
 [ FN  TP ]]

## 🔧 Model Architecture (Modified InceptionV3)
InceptionV3 (weights=None)
 └── FC Layer: 2048 → 1024 → 512 → 2 (Male/Female)


AuxLogits are ignored to stay compatible with offline weight files.
