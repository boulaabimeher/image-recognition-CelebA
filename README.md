# Image Recognition on CelebA (Gender Classification)

This repository implements **gender classification (Male / Female)** on the **CelebA dataset** using deep learning.
The project is designed to **run fully offline on HPC clusters** (e.g. CRIL), with no internet access during training or evaluation.

---

## 📂 Repository Structure

This repository contains **three branches**, each corresponding to a different framework or implementation:

| Branch | Description |
|------|------------|
| `main` | Project overview and documentation |
| `pytorch_version` | **PyTorch implementation (offline training on CRIL cluster)** |
| `tensorflow_version` | TensorFlow / Keras implementation |

> ⚠️ **All training on the CRIL cluster is done using the `pytorch_version` branch**

---

## 🧠 Task Description

- **Dataset**: CelebA
- **Task**: Binary classification — `Male` vs `Female`
- **Model**: InceptionV3 (trained offline)
- **Loss**: CrossEntropyLoss
- **Optimizer**: SGD
- **Metrics**:
  - Accuracy
  - F1-score
  - Confusion Matrix
  - ROC Curve (AUC)

---

## 🚀 Offline Training on CRIL Cluster (PyTorch)

### 1️⃣ Clone the repository (login node)

```bash
git clone <your-repo-url>
cd image-recognition-CelebA
git checkout pytorch_version
2️⃣ Dataset Preparation
Download CelebA once on a machine with internet access, then copy it to the cluster.

Expected structure:

text
Copy code
data/celeba/
├── img_align_celeba/
│   └── img_align_celeba/
│       ├── 000001.jpg
│       ├── 000002.jpg
│       └── ...
├── list_attr_celeba.csv
└── list_eval_partition.csv
Update paths in main.py if needed:

python
Copy code
main_folder = "../data/celeba/"
3️⃣ Python Environment (Offline)
Create a virtual environment and install packages offline (wheelhouse method):

bash
Copy code
python -m venv img-reco
source img-reco/bin/activate
pip install --no-index --find-links=wheelhouse -r requirements.txt
4️⃣ Offline Weights (InceptionV3)
Pretrained weights must be downloaded before running on the cluster:

text
Copy code
pytorch_version/
└── inception/
    └── inception_v3_weights.pth
The model is loaded without internet access:

python
Copy code
models.inception_v3(weights=None, aux_logits=False)
5️⃣ Training the Model
Submit the job using SLURM:

bash
Copy code
sbatch train.slurm
Or run interactively (if allowed):

bash
Copy code
python main.py
During training:

Progress is displayed with tqdm

Best model is saved automatically

text
Copy code
output/
└── best_model.pth
📊 Evaluation (No Retraining)
Evaluation does NOT retrain the model.

Run:

bash
Copy code
python eval.py
This will:

Load best_model.pth

Evaluate on the test split

Generate plots

Outputs:

text
Copy code
output/
├── confusion_matrix.png
├── roc_curve.png
└── best_model.pth
🛑 Important Design Choice
To avoid accidental retraining:

python
Copy code
if __name__ == "__main__":
    # training code
This ensures:

main.py trains only when executed directly

eval.py can safely import shared code

📈 Results
Typical performance after training:

Accuracy: ~99%

F1-score: ~0.99

Strong ROC-AUC

(Exact results may vary depending on training duration and hardware.)

🧪 Reproducibility
Fixed dataset splits using CelebA official partitions

Offline execution

Deterministic evaluation pipeline

🔬 Future Work
Multi-attribute classification

Bias and fairness analysis

Concept Bottleneck Models (CBMs)

Distributed training

Explainability (Grad-CAM, CAM)

👤 Author
Meher Boulaabi
Artificial Intelligence Scientist
Medical Image Analysis & Deep Learning

⭐ Acknowledgements
CelebA Dataset

