import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
from tensorflow.keras.utils import to_categorical

from keras.optimizers import SGD

from IPython.display import display, HTML
from PIL import Image
from io import BytesIO
import base64

plt.style.use("ggplot")

import tensorflow as tf

print(tf.__version__)

# ==============================
# Step 1: Data Exploration
# ==============================
main_folder = "data/celeba/"
images_folder = main_folder + "img_align_celeba/img_align_celeba/"

EXAMPLE_PIC = images_folder + "000506.jpg"

TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
NUM_EPOCHS = 20

# Load attributes dataset for each image
df_attr = pd.read_csv(main_folder + "list_attr_celeba.csv")
df_attr.set_index("image_id", inplace=True)
df_attr.replace(to_replace=-1, value=0, inplace=True)
df_attr.shape

# Display available attributes in CelebA dataset
for i, j in enumerate(df_attr.columns):
    print(i, j)

# Visualize an example image
img = load_img(EXAMPLE_PIC)
plt.figure(figsize=(6, 6))
plt.grid(False)
plt.imshow(img)
plt.savefig("output/example_image_plot.png", dpi=300, bbox_inches="tight")

# Display attributes of the example image
df_attr.loc[EXAMPLE_PIC.split("/")[-1]][["Smiling", "Male", "Young"]]

# Plot distribution of Male/Female attribute
plt.figure(figsize=(8, 6))
plt.title("Female or Male")
plt.title("Gender Distribution")
sns.countplot(y="Male", data=df_attr, color="c")
plt.savefig("output/female_male_countplot.png", dpi=300, bbox_inches="tight")

# ==============================
# Step 2: Dataset Partitioning
# ==============================
# Load partition info for train, validation, and test sets
df_partition = pd.read_csv(main_folder + "list_eval_partition.csv")
df_partition.head()

# Display sample counts per partition
df_partition["partition"].value_counts().sort_index()

# Merge partition info with gender attribute
df_partition.set_index("image_id", inplace=True)
df_par_attr = df_partition.join(df_attr["Male"], how="inner")
df_par_attr.head()


# ==============================
# Step 2.1: Data Loading and Preprocessing Functions
# ==============================
def load_reshape_img(fname):
    """Load an image and normalize pixel values."""
    img = load_img(fname)
    x = img_to_array(img) / 255.0
    x = x.reshape((1,) + x.shape)
    return x


def generate_df(partition, attr, num_samples):
    """
    Generate dataset for the specified partition and attribute.
    Parameters:
    partition: 0=train, 1=validation, 2=test
    attr: target attribute
    num_samples: number of samples to generate
    """
    # Sample balanced data for the attribute
    df_ = df_par_attr[
        (df_par_attr["partition"] == partition) & (df_par_attr[attr] == 0)
    ].sample(int(num_samples / 2))
    df_ = pd.concat(
        [
            df_,
            df_par_attr[
                (df_par_attr["partition"] == partition) & (df_par_attr[attr] == 1)
            ].sample(int(num_samples / 2)),
        ]
    )

    # For training and validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = to_categorical(df_[attr].values, num_classes=2)

    # For testing
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis=0)

            x_.append(im)
            y_.append(target[attr])

        # Convert lists to NumPy arrays
        x_ = np.concatenate(x_, axis=0)
        y_ = np.array(y_)

    return x_, y_


# ==============================
# Step 3: Data Augmentation
# ==============================
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Visualize augmented examples
img = load_img(EXAMPLE_PIC)
x = img_to_array(img) / 255.0
x = x.reshape((1,) + x.shape)

plt.figure(figsize=(20, 10))
plt.suptitle("Data Augmentation Examples", fontsize=28)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(3, 5, i + 1)
    plt.grid(False)
    plt.imshow(batch.reshape(218, 178, 3))
    if i == 9:
        break
    i += 1

plt.savefig("output/data_augmentation_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# ==============================
# Step 3.2: Create Data Generators
# ==============================
x_train, y_train = generate_df(0, "Male", TRAINING_SAMPLES)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
train_datagen.fit(x_train)

train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

x_valid, y_valid = generate_df(1, "Male", VALIDATION_SAMPLES)

# ==============================
# Step 4: Model Definition - Gender Classification
# ==============================
weights_path = "inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Load pretrained InceptionV3 model without top layers
inc_model = InceptionV3(
    weights=weights_path,
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
)
print("Number of layers:", len(inc_model.layers))

# Add custom classification layers
x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model_ = Model(inputs=inc_model.input, outputs=predictions)

# Freeze initial layers to retain pretrained features
for layer in model_.layers[:52]:
    layer.trainable = False

# Compile model
model_.compile(
    optimizer=SGD(learning_rate=0.0001, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Model checkpoint to save the best performing model
checkpointer = ModelCheckpoint(
    filepath="best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

# ==============================
# Step 5: Model Training
# ==============================
hist = model_.fit(
    train_generator,
    validation_data=(x_valid, y_valid),
    steps_per_epoch=TRAINING_SAMPLES // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[checkpointer],
    verbose=1,
)

# ==============================
# Step 6: Visualize Training Progress
# ==============================
plt.figure(figsize=(18, 4))
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss Over Epochs")
plt.savefig("output/loss_plot.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(18, 4))
plt.plot(hist.history["accuracy"], label="Train Accuracy")
plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Over Epochs")
plt.savefig("output/accuracy_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# ==============================
# Step 7: Model Evaluation
# ==============================
# Load best model weights
model_.load_weights("best_model.keras")

# Generate test set
x_test, y_test = generate_df(2, "Male", TEST_SAMPLES)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Make predictions and evaluate
model_predictions = np.argmax(model_.predict(x_test), axis=1)
test_accuracy = 100 * np.sum(model_predictions == y_test) / len(model_predictions)

print("Model Evaluation")
print("Test Accuracy: %.4f%%" % test_accuracy)
print("F1 Score:", f1_score(y_test, model_predictions))
