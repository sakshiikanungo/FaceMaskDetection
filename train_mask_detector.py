import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# Initializing the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Path to the directory where your images are stored
DIRECTORY = r"D:\college_project\dataset"

# Ensure that CATEGORIES matches the actual folder names
CATEGORIES = ["with_mask", "without_mask"]  # Replace with the actual folder names

print("Please wait while the images are being loaded...")

# Data list to store the images and the Labels list to store the corresponding labels (mask, no mask)
data = []
labels = []

# Image Preprocessing before feeding the images to the model
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)  # Creating full path to the directory of the images
    if os.path.exists(path):
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                # Load image with target size and convert to RGB if it has a palette
                image = load_img(img_path, target_size=(224, 224))

# Check if the image has transparency and convert if necessary
                if image.mode == 'P' and 'transparency' in image.info:  # Palette image with transparency
                    image = image.convert('RGBA')
                elif image.mode != 'RGB':  # Convert other non-RGB images to RGB
                    image = image.convert('RGB')

                image = img_to_array(image)  # Convert to numpy array format
                image = preprocess_input(image)  # Preprocess for MobileNetV2
                # Scale pixel intensities to [-1, 1]

                # Append image and corresponding label
                data.append(image)
                labels.append(category)

            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
    else:
        print(f"Category folder '{category}' not found in {DIRECTORY}")

# Ensure labels are converted to binary values (0 for 'with_mask', 1 for 'without_mask')
label_map = {CATEGORIES[0]: 0, CATEGORIES[1]: 1}
labels = [label_map[label] for label in labels]

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Convert data to a NumPy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split the dataset into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=np.argmax(labels, axis=1), random_state=42)

# Construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Load the MobileNetV2 network, excluding the fully connected layer at the top
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Construct the head of the model to be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Place the fully connected head on top of the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze all layers in the base model
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)  # Corrected initialization
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])  # Use categorical_crossentropy

# Train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Find the index of the label with the largest predicted probability for each image
predIdxs = np.argmax(predIdxs, axis=1)

# Show a nicely formatted classification report
print(classification_report(np.argmax(testY, axis=1), predIdxs, target_names=["with_mask", "without_mask"]))


# Save the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.keras", save_format="h5")

# Plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

