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

# ## Step 1: Data Exploration
# set variables
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


# ### Load the attributes of every picture
# File: list_attr_celeba.csv


# import the data set that include the attribute for each picture
df_attr = pd.read_csv(main_folder + "list_attr_celeba.csv")
df_attr.set_index("image_id", inplace=True)
df_attr.replace(to_replace=-1, value=0, inplace=True)  # replace -1 by 0
df_attr.shape


# ### List of the available attribute in the CelebA dataset
#
# 40 Attributes
# List of available attributes
for i, j in enumerate(df_attr.columns):
    print(i, j)


# ### Example of a picture in CelebA dataset
# 178 x 218 px


# Plot image
img = load_img(EXAMPLE_PIC)
plt.figure(figsize=(6, 6))
plt.grid(False)
plt.imshow(img)

# Save the figure
plt.savefig("output/example_image_plot.png", dpi=300, bbox_inches="tight")

df_attr.loc[EXAMPLE_PIC.split("/")[-1]][["Smiling", "Male", "Young"]]  # some attributes


# ### Distribution of the Attribute
# Female or Male?

plt.figure(figsize=(8, 6))
plt.title("Female or Male")
sns.countplot(y="Male", data=df_attr, color="c")
# Save to file
plt.savefig("output/female_male_countplot.png", dpi=300, bbox_inches="tight")

# ## Step 2: Split Dataset into Training, Validation and Test
# Recomended partition
df_partition = pd.read_csv(main_folder + "list_eval_partition.csv")
df_partition.head()


# display counter by partition
# 0 -> TRAINING
# 1 -> VALIDATION
# 2 -> TEST
df_partition["partition"].value_counts().sort_index()


# #### Join the partition and the attributes in the same data frame
# join the partition with the attributes
df_partition.set_index("image_id", inplace=True)
df_par_attr = df_partition.join(df_attr["Male"], how="inner")
df_par_attr.head()


# ### 2.1: Generate Partitions (Train, Validation, Test)


def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img) / 255.0
    x = x.reshape((1,) + x.shape)

    return x


def generate_df(partition, attr, num_samples):
    """
    partition
        0 -> train
        1 -> validation
        2 -> test

    """

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

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)

        # Replace deprecated np_utils.to_categorical
        y_ = to_categorical(df_[attr].values, num_classes=2)

    # for Test
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

        # Convert lists to numpy arrays (recommended)
        x_ = np.concatenate(x_, axis=0)
        y_ = np.array(y_)

    return x_, y_


# Generate image generator for data augmentation
datagen = ImageDataGenerator(
    # preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# load one image and reshape
img = load_img(EXAMPLE_PIC)
x = img_to_array(img) / 255.0
x = x.reshape((1,) + x.shape)

plt.figure(figsize=(20, 10))
plt.suptitle("Data Augmentation", fontsize=28)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(3, 5, i + 1)
    plt.grid(False)
    plt.imshow(batch.reshape(218, 178, 3))

    if i == 9:
        break
    i += 1

# Save the figure
plt.savefig("output/data_augmentation_plot.png", dpi=300, bbox_inches="tight")

plt.show()

# ### 3.2. Build Data Generators

# Train data
x_train, y_train = generate_df(0, "Male", TRAINING_SAMPLES)

# Train - Data Preparation - Data Augmentation with generators
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

train_generator = train_datagen.flow(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
)


# Validation Data
x_valid, y_valid = generate_df(1, "Male", VALIDATION_SAMPLES)

"""
# Validation - Data Preparation - Data Augmentation with generators
valid_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
)

valid_datagen.fit(x_valid)

validation_generator = valid_datagen.flow(
x_valid, y_valid,
)
"""


# With the data generator created and data for validation, we are ready to start modeling.


# ## Step 4: Build the Model - Gender Recognition


# ### 4.1. Set the Model


# Import InceptionV3 Model

# Path to local InceptionV3 weights
weights_path = "inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# ==============================
# LOAD PRETRAINED MODEL
# ==============================
inc_model = InceptionV3(
    weights=weights_path,  # use local weights
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
)

print("Number of layers:", len(inc_model.layers))

# ==============================
# ADD CUSTOM LAYERS
# ==============================

x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)


model_ = Model(inputs=inc_model.input, outputs=predictions)

# Freeze initial layers
for layer in model_.layers[:52]:
    layer.trainable = False

# ==============================
# COMPILE MODEL
# ==============================

model_.compile(
    optimizer=SGD(learning_rate=0.0001, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ==============================
# MODEL CHECKPOINT
# ==============================
checkpointer = ModelCheckpoint(
    filepath="best_model.keras",  # must end with .keras for full model save
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)


# ==============================
# TRAIN MODEL
# ==============================
# train_generator must be a generator or tf.keras.utils.Sequence
hist = model_.fit(
    train_generator,
    validation_data=(x_valid, y_valid),
    steps_per_epoch=TRAINING_SAMPLES // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[checkpointer],
    verbose=1,
)

# ==============================
# PLOT LOSS & ACCURACY
# ==============================
plt.figure(figsize=(18, 4))
plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="valid")
plt.legend()
plt.title("Loss Function")
plt.savefig("output/loss_plot.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(18, 4))
plt.plot(hist.history["accuracy"], label="train")
plt.plot(hist.history["val_accuracy"], label="valid")
plt.legend()
plt.title("Accuracy")
plt.savefig("output/accuracy_plot.png", dpi=300, bbox_inches="tight")
plt.show()


# ==============================
# EVALUATION
# ==============================
# Load the best saved model
model_.load_weights("best_model.keras")

# Generate test data (replace with your function)
x_test, y_test = generate_df(2, "Male", TEST_SAMPLES)

# Convert test data to NumPy array if needed
x_test = np.array(x_test)
y_test = np.array(y_test)

# Predictions
model_predictions = np.argmax(model_.predict(x_test), axis=1)

# Test accuracy
test_accuracy = 100 * np.sum(model_predictions == y_test) / len(model_predictions)
print("Model Evaluation")
print("Test accuracy: %.4f%%" % test_accuracy)
print("F1 score:", f1_score(y_test, model_predictions))
