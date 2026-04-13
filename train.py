import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import json

# ==========================================
# 1. Configuration & Parameters
# ==========================================
# Make sure your dataset is in the 'dataset' folder in the same directory as this script.
DATASET_DIR = 'dataset' 
IMG_SIZE = (224, 224)   # Image size required for the CNN
BATCH_SIZE = 32         # Number of images to process at once
EPOCHS = 15             # Number of times the model will run through the dataset

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
# We use ImageDataGenerator to load images from directories, normalize them,
# and perform data augmentation for the training set.
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0, 1]
    rotation_range=20,       # Randomly rotate images
    zoom_range=0.2,          # Randomly zoom in/out
    horizontal_flip=True,    # Randomly flip images horizontally
    validation_split=0.2     # Set aside 20% of the data for validation
)

print("Loading training data...")
# Uses 80% of data for training automatically
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Used for multi-class classification
    subset='training'         # Specify this is the training set
)

print("Loading validation data...")
# Uses 20% of data for validation automatically
validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'       # Specify this is the validation set
)

# Save the class dictionary to use it later during prediction
class_indices = train_generator.class_indices
# Invert the dictionary to map index to class name
labels_map = {v: k for k, v in class_indices.items()}

# Save the label map to a JSON file so predict.py can load it
with open('class_labels.json', 'w') as f:
    json.dump(labels_map, f)
print("Class labels saved to 'class_labels.json':", labels_map)

# ==========================================
# 3. Building the CNN Model
# ==========================================
model = Sequential([
    # 1st Convolutional Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # 2nd Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # 3rd Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten the features to feed into Dense Layers
    Flatten(),
    
    # Fully Connected Layer
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout to prevent overfitting by ignoring 50% of neurons randomly
    
    # Output Layer (Number of nodes must match number of classes)
    Dense(len(class_indices), activation='softmax') # softmax for multi-class classification
])

# Compile the model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 4. Training the Model
# ==========================================
print("\nStarting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# ==========================================
# 5. Saving the Model
# ==========================================
model.save('model.h5')
print("\nModel trained successfully and saved as 'model.h5'")

# ==========================================
# 6. Plotting Accuracy and Loss Graphs
# ==========================================
# Plot training & validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_plot.png')
print("Training plot saved as 'training_plot.png'.")
plt.show()
