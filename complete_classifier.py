import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# =======================================================
# 1. CONFIGURATION (Lightweight for Low-End Laptop)
# =======================================================
DATASET_DIR = "Vegetable_Dataset"
IMG_SIZE = (128, 128)   # Small size for low-end systems
BATCH_SIZE = 16         # Small batch size to save memory
EPOCHS = 8              # Fewer epochs for fast training
MODEL_NAME = "vegetable_model.h5"

# =======================================================
# 2. LOAD DATA WITHOUT CHANGING THE FOLDER STRUCTURE
# =======================================================
# Since your dataset has a 2-level structure (Vegetable/Disease/), 
# we'll map the exact file paths and their combined labels dynamically!
filepaths = []
labels = []

if os.path.exists(DATASET_DIR):
    for vegetable_name in os.listdir(DATASET_DIR):
        veg_path = os.path.join(DATASET_DIR, vegetable_name)
        if os.path.isdir(veg_path):
            for disease_name in os.listdir(veg_path):
                disease_path = os.path.join(veg_path, disease_name)
                if os.path.isdir(disease_path):
                    for img_name in os.listdir(disease_path):
                        full_img_path = os.path.join(disease_path, img_name)
                        # Combines both names so model knows Tomato Blight from Potato Blight
                        label = f"{vegetable_name}_{disease_name}"
                        filepaths.append(full_img_path)
                        labels.append(label)

    # DataFrame allows the Generator to read from any custom folder structure instantly
    df = pd.DataFrame({"filepath": filepaths, "label": labels})
    print(f"Total images found: {len(df)}")
else:
    df = pd.DataFrame(columns=["filepath", "label"])
    print(f"Warning: Folder '{DATASET_DIR}' not found. Cannot train.")

# =======================================================
# 3. TRAINING FUNCTION
# =======================================================
def train_model():
    if len(df) == 0:
        return None, None
        
    print("\n--- Prepping Data ---")
    # Normalizing via 1./255 prevents the "predicting same output every time" issue!
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=15,     # Rotate images lightly
        horizontal_flip=True,  # Flips images left-right
        validation_split=0.2   # Set aside 20% data for validation automatically
    )

    # Training Data stream
    train_gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    # Validation Data stream
    val_gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )
    
    # Save class assignments for future prediction
    class_indices = train_gen.class_indices
    class_dict = {v: k for k, v in class_indices.items()}

    print("\n--- Building Small CNN Model ---")
    model = Sequential([
        # 1st Block (16 Filters, minimal impact on RAM)
        Conv2D(16, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2,2),
        
        # 2nd Block (32 Filters)
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        # Flattening 2D to 1D
        Flatten(),
        
        # Final layers
        Dense(64, activation='relu'),                 # Small layer
        Dropout(0.3),                                 # Drop 30% neurons to prevent overfitting
        Dense(len(class_dict), activation='softmax')  # Softmax generates clear distinct probabilities
    ])

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model.summary()

    print("\n--- Starting Training ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    model.save(MODEL_NAME)
    print(f"\nModel efficiently trained and saved to {MODEL_NAME}")
    
    # Plot performance results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('training_graph.png')
    print("Saved graph as training_graph.png")
    
    return model, class_dict

# =======================================================
# 4. SINGLE IMAGE PREDICTION FUNCTION
# =======================================================
def predict_image(img_path, trained_model, class_dictionary):
    """
    Evaluates one single image using the saved CNN model.
    """
    # 1. Load image and resize to (128, 128)
    img = load_img(img_path, target_size=IMG_SIZE)
    
    # 2. Convert to numbers
    img_array = img_to_array(img)
    
    # 3. Add batch-wrapper dimension -> from (128,128,3) to (1,128,128,3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. CRITICAL: Rescale exactly like we did in ImageDataGenerator
    img_array = img_array / 255.0
    
    # 5. Ask model to guess
    probs = trained_model.predict(img_array, verbose=0)[0] 
    
    # 6. Extract the disease with highest confidence
    max_index = np.argmax(probs)
    predicted_disease = class_dictionary[max_index]
    confidence_percent = probs[max_index] * 100
    
    return predicted_disease, confidence_percent

# =======================================================
# 5. EXECUTION BLOCK
# =======================================================
if __name__ == "__main__":
    
    print("="*40)
    print(" MULTI-TIER VEGETABLE DISEASE AI")
    print("="*40)
    
    model = None
    
    # Run training if model file does not exist
    if not os.path.exists(MODEL_NAME):
        model, class_map = train_model()
    else:
        print(f"Found existing '{MODEL_NAME}'. Skipping training to save time.")
        print("-> Tip: Delete the .h5 file if you want to rebuild it from scratch.\n")
        
        model = load_model(MODEL_NAME)
        
        # We recreate the class mapping dictionary dynamically based on folder names
        if len(df) > 0:
            unique_labels = sorted(df['label'].unique())
            class_map = {i: c for i, c in enumerate(unique_labels)}
        else:
            print(f"Error: Could not locate '{DATASET_DIR}'.")
            exit()
    
    # Execute a final test prediction to prove it works dynamically!
    if model is not None and len(df) > 0:
        print("\n--- Running Live Prediction Test ---")
        
        # We grab one random image path from the dataset
        random_test_img = df['filepath'].sample(1).values[0]
        
        disease, conf = predict_image(random_test_img, model, class_map)
        
        print("\n-- RESULTS --")
        print(f"Test Image: {os.path.basename(random_test_img)}")
        print(f"Classified: {disease}")
        print(f"Confidence: {conf:.2f}%")
        print("-------------")
