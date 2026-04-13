"""
=============================================================================
train_model.py — Fast CPU CNN Model Training Script
=============================================================================

Run this ONCE before deploying to train both CNN models:
    python train_model.py

Models saved to: 
    models/disease_model.keras
    models/container_model.keras

Features optimized for CPU:
    - Custom light CNN (replaces MobileNetV2)
    - 128x128 image size for huge speedup
    - Reduced augmentation logic
    - Multithreaded data generators
=============================================================================
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF log noise

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print(f"TensorFlow {tf.__version__} ready")
except ImportError:
    print("ERROR: Install TensorFlow first:  pip install tensorflow")
    sys.exit(1)

os.makedirs('models', exist_ok=True)

# ─── Shared Config ─────────────────────────────────────────────────────────────
IMG_SIZE   = (128, 128)  # Reduced from 224x224 (73% fewer pixels directly speeds up CPU math)
BATCH_SIZE = 16          # Reduced from 32 to prevent CPU cache thrashing
EPOCHS     = 10          # Fewer epochs needed since model is smaller
LR_INITIAL = 1e-3

def build_simple_cnn(num_classes: int) -> keras.Model:
    """
    Lightweight Custom CNN designed for fast CPU training.
    Replaces MobileNetV2 which is too heavy for standard CPUs.
    """
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),                  # Regularisation
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(LR_INITIAL),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_data_generators(dataset_path: str, img_size=IMG_SIZE, batch=BATCH_SIZE):
    """Create training and validation data generators with fast CPU augmentation."""
    # Simplified augmentation: Ultra-fast CPU processing
    train_gen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_data = train_gen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch,
        class_mode='categorical',
        subset='training',
        shuffle=True,
    )
    val_data = train_gen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
    )
    return train_data, val_data


def train_disease_model():
    """Train the plant disease detection CNN."""
    dataset_path = 'dataset/disease'

    if not os.path.exists(dataset_path):
        print(f"\n⚠️  Dataset not found at: {dataset_path}")
        return False

    print("\n" + "="*60)
    print("TRAINING DISEASE DETECTION MODEL")
    print("="*60)

    train_data, val_data = get_data_generators(dataset_path)
    num_classes = len(train_data.class_indices)
    print(f"Found {num_classes} disease classes: {list(train_data.class_indices.keys())}")
    print(f"Training samples: {train_data.samples} | Validation: {val_data.samples}")

    model = build_simple_cnn(num_classes)
    print(f"\nModel parameters: {model.count_params():,}")

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint('models/disease_model_best.keras', save_best_only=True, verbose=1),
    ]

    print("\n--- Training Model ---")
    model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        verbose=1
    )

    # Save final model
    model.save('models/disease_model.keras')
    print("\n✅ Disease model saved to models/disease_model.keras")

    # Save class label mapping
    with open('models/disease_labels.json', 'w') as f:
        json_map = {str(v): k for k, v in train_data.class_indices.items()}
        import json; json.dump(json_map, f, indent=2)
    print("✅ Disease labels saved to models/disease_labels.json")
    return True


def train_container_model():
    """Train the container size classification CNN."""
    dataset_path = 'dataset/container'

    if not os.path.exists(dataset_path):
        return False

    print("\n" + "="*60)
    print("TRAINING CONTAINER SIZE CLASSIFICATION MODEL")
    print("="*60)

    train_data, val_data = get_data_generators(dataset_path)
    num_classes = len(train_data.class_indices)

    model = build_simple_cnn(num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2, verbose=1),
    ]

    model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        verbose=1
    )
    
    model.save('models/container_model.keras')
    print("\n✅ Container model saved to models/container_model.keras")

    with open('models/container_labels.json', 'w') as f:
        import json; json.dump({str(v): k for k, v in train_data.class_indices.items()}, f, indent=2)
    return True


def evaluate_models():
    """Print evaluation summary of both saved models."""
    import json
    for name, path, label_path in [
        ("Disease", "models/disease_model.keras", "models/disease_labels.json"),
        ("Container", "models/container_model.keras", "models/container_labels.json"),
    ]:
        if os.path.exists(path):
            model = keras.models.load_model(path)
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"\n{name} Model:")
            print(f"  Path: {path}")
            print(f"  Size: {size_mb:.1f} MB")
            print(f"  Parameters: {model.count_params():,}")
        else:
            print(f"\n{name} Model: NOT FOUND — app will use heuristic fallback")


if __name__ == '__main__':
    print("AI Kitchen Garden — Fast CPU CNN Trainer")

    d = train_disease_model()
    c = train_container_model()
    evaluate_models()

    if d or c:
        print("\n✅ Training complete! Models saved in models/ folder.")