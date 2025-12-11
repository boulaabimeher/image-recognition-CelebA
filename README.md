<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Gender Recognition on CelebA Dataset</title>
</head>
<body>
    <h1>Gender Recognition on CelebA Dataset</h1>

    <h2>Project Overview</h2>
    <p>
        This project is focused on learning how to run deep learning code in a cluster environment and 
        to practice handling the <strong>CelebA</strong> dataset for image-based gender classification. 
        The original code was outdated and written for an earlier Python version. 
        I have corrected all functions and attributes to make it fully compatible with <strong>Python 3.13</strong>.
    </p>

    <h2>Project Goals</h2>
    <ul>
        <li>Understand how to prepare and handle the CelebA dataset for machine learning tasks.</li>
        <li>Learn how to run deep learning models on cluster systems.</li>
        <li>Update and refactor old code to modern Python standards.</li>
        <li>Implement gender recognition using a pretrained <strong>InceptionV3</strong> model with custom layers.</li>
    </ul>

    <h2>Dataset</h2>
    <p>
        The project uses the <strong>CelebA dataset</strong>, a large-scale face attributes dataset with more than 200,000 celebrity images, 
        each annotated with 40 binary attributes (such as gender, smiling, etc.). 
        This project focuses specifically on the <strong>gender (Male/Female)</strong> attribute.
    </p>
    <p>
        <strong>Dataset Structure:</strong>
        <ul>
            <li><code>img_align_celeba/</code>: Contains all aligned celebrity images.</li>
            <li><code>list_attr_celeba.csv</code>: Contains attributes for each image.</li>
            <li><code>list_eval_partition.csv</code>: Contains partitioning information for train, validation, and test sets.</li>
        </ul>
    </p>

    <h2>Dependencies</h2>
    <ul>
        <li>Python 3.13</li>
        <li>TensorFlow</li>
        <li>Keras</li>
        <li>NumPy</li>
        <li>Pandas</li>
        <li>OpenCV</li>
        <li>Matplotlib</li>
        <li>Seaborn</li>
        <li>Pillow (PIL)</li>
        <li>scikit-learn</li>
    </ul>

    <h2>Usage Instructions</h2>
    <ol>
        <li>Clone the repository.</li>
        <li>Download the CelebA dataset and place it in the <code>data/celeba/</code> directory.</li>
        <li>Ensure that the pretrained InceptionV3 weights file (<code>inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5</code>) is placed in the <code>inception/</code> folder.</li>
        <li>Install required dependencies using <code>pip install -r requirements.txt</code> (create this file with the packages listed above).</li>
        <li>Run the main script to train and evaluate the gender recognition model.</li>
        <li>Outputs such as training plots and the best model weights will be saved automatically in the <code>output/</code> folder.</li>
    </ol>

    <h2>Project Structure</h2>
    <ul>
        <li><code>main.py</code>: The main script containing the entire training and evaluation pipeline.</li>
        <li><code>data/celeba/</code>: Folder containing CelebA images and attribute CSV files.</li>
        <li><code>inception/</code>: Folder containing the InceptionV3 pretrained weights.</li>
        <li><code>output/</code>: Folder where all generated plots and model weights are saved.</li>
    </ul>

    <h2>Results</h2>
    <p>
        The script trains a custom classifier on top of a pretrained InceptionV3 model. 
        It generates plots for data augmentation examples, training/validation loss, and accuracy curves. 
        The model checkpoint saves the best model based on validation loss, and the final evaluation reports test accuracy and F1 score.
    </p>

    <h2>References & Credits</h2>
    <p>
        This project was inspired by an existing Kaggle notebook, which I modified and updated to Python 3.13 standards: 
        <a href="https://www.kaggle.com/code/bmarcos/image-recognition-gender-detection-inceptionv3" target="_blank">
            Gender Detection with InceptionV3 on Kaggle
        </a>.
    </p>

    <h2>License</h2>
    <p>
        This project is open-source and free to use for learning and research purposes.
    </p>
</body>
</html>
