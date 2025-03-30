import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Path to flat dataset
DATASET_DIR = "plant_dataset"

# Prepare image and label lists
images = []
labels = []

# Load and label images
for filename in os.listdir(DATASET_DIR):
    if filename.endswith(".jpeg"):
        label = filename.replace(".jpeg", "")  # e.g. okra_stage_1
        img_path = os.path.join(DATASET_DIR, filename)

        # Load image and preprocess
        img = Image.open(img_path).resize((150, 150)).convert('RGB')
        img_array = np.array(img) / 255.0
        images.append(img_array)
        labels.append(label)

# Encode text labels into integers
label_names = sorted(list(set(labels)))
label_to_index = {name: idx for idx, name in enumerate(label_names)}
y = np.array([label_to_index[label] for label in labels])
X = np.array(images)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(label_names))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(label_names))

# Build a simple CNN model
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_names), activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Save model and labels
model.save('plant_growth_model.h5')

with open('label_names.txt', 'w') as f:
    for label in label_names:
        f.write(f"{label}\n")
