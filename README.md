# Gender Recognition on CelebA Dataset

## Project Overview
This project is focused on learning how to run deep learning code in a cluster environment and to practice handling the **CelebA** dataset for image-based gender classification. The original code was outdated and written for an earlier Python version. I have corrected all functions and attributes to make it fully compatible with **Python 3.13**.

## Project Goals
- Understand how to prepare and handle the CelebA dataset for machine learning tasks.
- Learn how to run deep learning models on cluster systems.
- Update and refactor old code to modern Python standards.
- Implement gender recognition using a pretrained **InceptionV3** model with custom layers.

## Dataset
The project uses the **CelebA dataset**, a large-scale face attributes dataset with more than 200,000 celebrity images, each annotated with 40 binary attributes (such as gender, smiling, etc.). This project focuses specifically on the **gender (Male/Female)** attribute.

**Dataset Structure:**
- `img_align_celeba/`: Contains all aligned celebrity images.
- `list_attr_celeba.csv`: Contains attributes for each image.
- `list_eval_partition.csv`: Contains partitioning information for train, validation, and test sets.

## Dependencies
- Python 3.13
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- Pillow (PIL)
- scikit-learn

## Usage Instructions
1. Clone the repository.
2. Download the CelebA dataset and place it in the `data/celeba/` directory.
3. Ensure that the pretrained InceptionV3 weights file (`inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5`) is placed in the `inception/` folder.
4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
