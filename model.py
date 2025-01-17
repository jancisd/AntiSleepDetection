import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D
import tensorflow as tf

# Define data generator
def generator(dir, gen=ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=32, target_size=(24, 24), class_mode='categorical'):
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode='grayscale',  # Grayscale images
        class_mode=class_mode,
        target_size=target_size
    )

# Batch size and image dimensions
BS = 32
TS = (24, 24)

# Dataset paths
train_batch = generator('dataset_new/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('dataset_new/test', shuffle=True, batch_size=BS, target_size=TS)

# Steps per epoch and validation steps
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print("Steps per epoch:", SPE, "Validation steps:", VS)

# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_batch,
    validation_data=valid_batch,
    epochs=15,
    steps_per_epoch=SPE,
    validation_steps=VS
)

# Export the model to TensorFlow SavedModel format
model.export('models/cnnCat2_saved_model')
print("Model exported successfully!")
