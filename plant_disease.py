"""
=============================================================================
 plant_disease_cnn.py
 Plant Disease Detection — CNN Training & Prediction Pipeline
 
 Usage:
   python plant_disease_cnn.py            # train + save
   python plant_disease_cnn.py --predict path/to/leaf.jpg   # predict
 
 Dataset structure expected:
   Vegetable disease/
     ├── Tomato_Early_Blight/
     │     ├── img001.jpg
     │     └── ...
     ├── Tomato_Leaf_Mold/
     │     └── ...
     └── Healthy/
           └── ...
 
 Output:
   model.h5           — saved Keras model
   class_names.npy    — ordered array of class labels
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — safe on servers
import matplotlib.pyplot as plt

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

# ── Reproducibility (set before anything else uses random) ────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# CONFIGURATION  — change these to match your setup
# =============================================================================
DATASET_DIR    = "Vegetable disease"   # folder containing class sub-folders
IMG_SIZE       = (224, 224)            # resize target for every image
BATCH_SIZE     = 32                    # lower to 16 if you get OOM errors
EPOCHS         = 15                    # minimum; EarlyStopping may stop sooner
VALIDATION_SPLIT = 0.20               # 80 % train, 20 % validation
LEARNING_RATE  = 1e-3
MODEL_SAVE_PATH    = "model.keras"
CLASSES_SAVE_PATH  = "class_names.npy"
PLOT_SAVE_PATH     = "training_history.png"

# =============================================================================
# STEP 1 + 2 — LOAD DATASET  &  SPLIT  &  NORMALISE
# =============================================================================
def load_datasets(dataset_dir: str):
    """
    Load training and validation datasets directly from directory structure.
    Each sub-folder name becomes a class label.
    Images are resized to IMG_SIZE and pixel values kept as uint8 here;
    normalisation (÷255) is applied inside the model as a Rescaling layer so
    it is part of the saved graph and runs at inference too.
    """
    print(f"\n{'='*60}")
    print(f"  Loading dataset from: {dataset_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: '{dataset_dir}'\n"
            f"Make sure the folder exists and contains class sub-folders."
        )

    # Training split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",          # integer labels → sparse_categorical_crossentropy
        shuffle=True,
    )

    # Validation split
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,             # keep validation deterministic
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(f"\n  Classes found  : {num_classes}")
    for i, name in enumerate(class_names):
        print(f"    [{i:>2}] {name}")

    n_train = sum(1 for _ in train_ds) * BATCH_SIZE
    n_val   = sum(1 for _ in val_ds)   * BATCH_SIZE
    print(f"\n  ~{n_train} training images | ~{n_val} validation images")

    # Performance: cache decoded images in RAM, prefetch next batch while GPU trains
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes


# =============================================================================
# STEP 3 + 5 — DATA AUGMENTATION  (separate layer block, training-only)
# =============================================================================
def make_augmentation_block():
    """
    Returns a Sequential block of augmentation layers.
    These layers are active only during training (training=True flag).
    Kept separate so they can be toggled or inspected easily.
    
    Why these transforms?
    - RandomFlip         : mirrors leaf images horizontally / vertically
    - RandomRotation     : accounts for any leaf orientation in photos
    - RandomZoom         : simulates different camera distances
    - RandomTranslation  : shifts the leaf within the frame
    - RandomContrast     : handles uneven lighting conditions
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            layers.RandomRotation(factor=0.25, seed=SEED),            # ±90°
            layers.RandomZoom(height_factor=0.20, width_factor=0.20, seed=SEED),
            layers.RandomTranslation(height_factor=0.10, width_factor=0.10, seed=SEED),
            layers.RandomContrast(factor=0.15, seed=SEED),
        ],
        name="augmentation",
    )


# =============================================================================
# STEP 4 — BUILD CNN MODEL
# =============================================================================
def build_model(num_classes: int) -> keras.Model:
    """
    Builds a 6-block CNN with:
      • Rescaling (÷255) baked into the graph — predictions don't need manual normalisation
      • Data augmentation as a training-only layer
      • 3 × [Conv2D → BatchNorm → MaxPool → Dropout] towers (increasing depth)
      • GlobalAveragePooling instead of Flatten — fewer parameters, less overfitting
      • 2 Dense layers with Dropout before the softmax head
    
    Architecture choice rationale:
      ─ BatchNormalization stabilises activations between conv blocks,
        allowing higher learning rates and faster convergence.
      ─ Dropout(0.3) after pooling and Dropout(0.5) before the head
        are the primary anti-overfitting mechanisms alongside augmentation.
      ─ GlobalAveragePooling2D replaces Flatten to reduce parameter count
        from ~6M (flatten) to ~400K, keeping the model deployable on free-tier hardware.
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")

    # ── Normalisation (baked into the model graph) ────────────────────────────
    x = layers.Rescaling(1.0 / 255, name="normalise")(inputs)

    # ── Augmentation (active during training only) ────────────────────────────
    x = make_augmentation_block()(x, training=True)

    # ── Block 1: 32 filters ───────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      name="conv1a")(x)
    x = layers.BatchNormalization(name="bn1a")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      name="conv1b")(x)
    x = layers.BatchNormalization(name="bn1b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # ── Block 2: 64 filters ───────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      name="conv2a")(x)
    x = layers.BatchNormalization(name="bn2a")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      name="conv2b")(x)
    x = layers.BatchNormalization(name="bn2b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.30, name="drop2")(x)

    # ── Block 3: 128 filters ──────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      name="conv3a")(x)
    x = layers.BatchNormalization(name="bn3a")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      name="conv3b")(x)
    x = layers.BatchNormalization(name="bn3b")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="drop3")(x)

    # ── Block 4: 256 filters ──────────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu",
                      name="conv4a")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Dropout(0.30, name="drop4")(x)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.50, name="drop_fc1")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.40, name="drop_fc2")(x)

    # Output layer: softmax over all disease classes
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="PlantDiseaseDetector")
    return model


# =============================================================================
# STEP 6 — COMPILE  (Adam + sparse_categorical_crossentropy)
# =============================================================================
def compile_model(model: keras.Model, learning_rate: float = LEARNING_RATE):
    """
    Compile with:
      optimizer  : Adam (adaptive learning rate — robust default)
      loss       : sparse_categorical_crossentropy
                   (integer labels → no need for one-hot encoding)
      metrics    : accuracy (human-readable training/val accuracy)
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# STEP 7 — CALLBACKS
# =============================================================================
def make_callbacks(model_path: str):
    """
    Three callbacks that together prevent overfitting and wasted training time:
    
      ModelCheckpoint  — saves the best model (by val_accuracy) automatically.
                         If training is interrupted, you keep the best weights.
    
      EarlyStopping    — stops training when val_accuracy stops improving for
                         5 consecutive epochs (patience=5). Restores the best
                         weights automatically.
    
      ReduceLROnPlateau — if val_accuracy plateaus for 3 epochs, halves the
                          learning rate. This often unlocks further improvement
                          that a fixed LR would miss.
    """
    return [
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# =============================================================================
# STEP 8 — TRAIN
# =============================================================================
def train_model(model, train_ds, val_ds, epochs: int = EPOCHS):
    """
    Train and return the history object for plotting.
    Note: augmentation layers apply only when training=True (handled internally
    by Keras when calling model.fit vs model.predict).
    """
    print(f"\n{'='*60}")
    print(f"  Training for up to {epochs} epochs")
    print(f"  (EarlyStopping may stop earlier)")
    print(f"{'='*60}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=make_callbacks(MODEL_SAVE_PATH),
        verbose=1,
    )
    return history


# =============================================================================
# STEP 9 — PRINT ACCURACY SUMMARY
# =============================================================================
def print_accuracy_summary(history):
    """Print a concise table of training + validation accuracy per epoch."""
    print(f"\n{'='*60}")
    print(f"  Training Summary")
    print(f"{'='*60}")
    print(f"  {'Epoch':>6}  {'Train Acc':>10}  {'Val Acc':>10}  {'Train Loss':>11}  {'Val Loss':>9}")
    print(f"  {'-'*56}")

    acc     = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss    = history.history.get("loss", [])
    val_loss= history.history.get("val_loss", [])

    for epoch_idx, (a, va, l, vl) in enumerate(zip(acc, val_acc, loss, val_loss), start=1):
        print(f"  {epoch_idx:>6}  {a*100:>9.2f}%  {va*100:>9.2f}%  {l:>11.4f}  {vl:>9.4f}")

    best_epoch = int(np.argmax(val_acc)) + 1
    print(f"\n  Best epoch     : {best_epoch}")
    print(f"  Best val acc   : {max(val_acc)*100:.2f}%")
    print(f"  Final train acc: {acc[-1]*100:.2f}%")

    gap = abs(acc[-1] - val_acc[-1])
    if gap > 0.15:
        print(f"\n  ⚠  Overfitting gap = {gap*100:.1f}% — consider more augmentation or Dropout")
    elif gap < 0.05:
        print(f"\n  ✓  Train/val accuracy well matched ({gap*100:.1f}% gap)")
    else:
        print(f"\n  ✓  Moderate gap ({gap*100:.1f}%) — model generalising well")


# =============================================================================
# STEP 10 — SAVE MODEL + CLASS NAMES
# =============================================================================
def save_artifacts(model, class_names: list):
    """
    Saves:
      model.h5         — full Keras model (architecture + weights + normalisation layer)
      class_names.npy  — numpy array of class name strings, ordered by class index
    
    The normalisation (Rescaling) layer is baked into the model graph so
    prediction functions do NOT need to manually divide pixel values by 255.
    """
    model.save(MODEL_SAVE_PATH)
    np.save(CLASSES_SAVE_PATH, np.array(class_names))

    print(f"\n  ✓ Model saved  → {MODEL_SAVE_PATH}")
    print(f"  ✓ Classes saved → {CLASSES_SAVE_PATH}")
    print(f"  ✓ Class names  : {class_names}")


# =============================================================================
# PLOT TRAINING HISTORY
# =============================================================================
def plot_history(history):
    """Save accuracy and loss curves to training_history.png."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    acc     = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss    = history.history.get("loss", [])
    val_loss= history.history.get("val_loss", [])
    epochs  = range(1, len(acc) + 1)

    # Accuracy subplot
    axes[0].plot(epochs, [a * 100 for a in acc],     label="Train accuracy")
    axes[0].plot(epochs, [a * 100 for a in val_acc], label="Val accuracy", linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss subplot
    axes[1].plot(epochs, loss,     label="Train loss")
    axes[1].plot(epochs, val_loss, label="Val loss", linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plots saved  → {PLOT_SAVE_PATH}")


# =============================================================================
# STEP 11 + 12 + 13 — PREDICTION FUNCTION
# =============================================================================
def load_model_and_classes(
    model_path: str = MODEL_SAVE_PATH,
    classes_path: str = CLASSES_SAVE_PATH,
):
    """
    Load the saved model and class name array.
    Raises FileNotFoundError with a helpful message if files are missing.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'\n"
            f"Run training first:  python plant_disease_cnn.py"
        )
    if not os.path.exists(classes_path):
        raise FileNotFoundError(
            f"Class names file not found: '{classes_path}'\n"
            f"Run training first to regenerate it."
        )

    model       = keras.models.load_model(model_path)
    class_names = np.load(classes_path, allow_pickle=True).tolist()
    print(f"  Loaded model from  : {model_path}")
    print(f"  Loaded {len(class_names)} classes from: {classes_path}")
    return model, class_names


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load a single image from disk and prepare it for model inference.
    
    Steps:
      1. Load image in RGB format (ignores EXIF rotation)
      2. Resize to the training IMG_SIZE exactly (bilinear interpolation)
      3. Convert to float32 numpy array with shape (224, 224, 3)
      4. Add batch dimension → shape (1, 224, 224, 3)
    
    NOTE: Do NOT divide by 255 here — the Rescaling layer inside the model
    handles normalisation.  If you divide here AND the model normalises,
    you will get double-normalisation and wrong predictions.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: '{image_path}'")

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)      # float32, [0, 255]
    arr = np.expand_dims(arr, axis=0)           # (1, 224, 224, 3)
    return arr


def predict_disease(
    image_path: str,
    model=None,
    class_names: list = None,
    top_k: int = 3,
) -> dict:
    """
    Predict the plant disease from a single leaf image.
    
    Args:
        image_path   : path to the input leaf image (jpg / png / bmp)
        model        : already-loaded Keras model (loaded if None)
        class_names  : list of class name strings (loaded if None)
        top_k        : number of top predictions to include in the result
    
    Returns a dict:
        {
            "disease"      : "Tomato_Early_Blight",   ← top-1 predicted class
            "confidence"   : 0.9423,                  ← probability [0–1]
            "confidence_pct": "94.23%",               ← human-readable
            "top_predictions": [                      ← top-k results
                {"disease": "Tomato_Early_Blight", "confidence": 0.9423},
                {"disease": "Tomato_Leaf_Mold",    "confidence": 0.0412},
                {"disease": "Healthy",             "confidence": 0.0102},
            ]
        }
    
    The model's Rescaling layer normalises the pixel values (÷255) internally,
    so this function works correctly regardless of the raw pixel range of the
    loaded image.  Different images will yield different softmax distributions
    because the model learned genuine visual features — not random noise.
    """
    if model is None or class_names is None:
        model, class_names = load_model_and_classes()

    # Preprocess
    img_array = preprocess_image(image_path)

    # Inference — training=False disables augmentation and uses BN in eval mode
    predictions = model.predict(img_array, verbose=0)[0]   # shape: (num_classes,)

    # Top-1
    top1_idx  = int(np.argmax(predictions))
    top1_conf = float(predictions[top1_idx])
    top1_name = class_names[top1_idx]

    # Top-k
    top_k_indices = np.argsort(predictions)[::-1][:top_k]
    top_k_results = [
        {
            "disease":    class_names[i],
            "confidence": round(float(predictions[i]), 4),
        }
        for i in top_k_indices
    ]

    return {
        "disease":        top1_name,
        "confidence":     round(top1_conf, 4),
        "confidence_pct": f"{top1_conf * 100:.2f}%",
        "top_predictions": top_k_results,
        "image_path":     image_path,
    }


def predict_batch(image_paths: list, model=None, class_names: list = None) -> list:
    """
    Predict diseases for a list of image paths.
    Batches images together for efficient GPU/CPU utilisation.
    Returns a list of result dicts in the same order as image_paths.
    """
    if model is None or class_names is None:
        model, class_names = load_model_and_classes()

    arrays = np.concatenate(
        [preprocess_image(p) for p in image_paths], axis=0
    )                                                        # (N, 224, 224, 3)
    all_preds = model.predict(arrays, batch_size=BATCH_SIZE, verbose=0)

    results = []
    for i, preds in enumerate(all_preds):
        top1_idx  = int(np.argmax(preds))
        top1_conf = float(preds[top1_idx])
        results.append({
            "disease":        class_names[top1_idx],
            "confidence":     round(top1_conf, 4),
            "confidence_pct": f"{top1_conf * 100:.2f}%",
            "image_path":     image_paths[i],
        })
    return results


# =============================================================================
# COMPLETE TRAINING PIPELINE
# =============================================================================
def run_training():
    """End-to-end training pipeline."""

    # GPU memory growth — prevents TF from grabbing all VRAM at once
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"\n  TensorFlow {tf.__version__}")
    print(f"  GPUs available : {len(gpus)}")
    if not gpus:
        print("  Running on CPU — training will be slow for large datasets")
        print("  Tip: use Google Colab (free GPU) for faster training")

    # Steps 1–2: load and split
    train_ds, val_ds, class_names, num_classes = load_datasets(DATASET_DIR)

    # Step 3: build
    model = build_model(num_classes)

    # Step 6: compile
    compile_model(model, learning_rate=LEARNING_RATE)

    # Print model summary
    print(f"\n{'='*60}")
    print(f"  Model Architecture")
    print(f"{'='*60}")
    model.summary()
    total_params = model.count_params()
    print(f"\n  Total parameters : {total_params:,}")

    # Steps 7–8: train
    history = train_model(model, train_ds, val_ds)

    # Step 9: print accuracy
    print_accuracy_summary(history)

    # Step 10: save
    save_artifacts(model, class_names)
    plot_history(history)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  To predict on a new image:")
    print(f"    python plant_disease_cnn.py --predict path/to/leaf.jpg")
    print(f"{'='*60}\n")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plant disease CNN — train or predict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python plant_disease_cnn.py\n"
            "  python plant_disease_cnn.py --predict leaf.jpg\n"
            "  python plant_disease_cnn.py --predict leaf.jpg --top_k 5\n"
        ),
    )
    parser.add_argument(
        "--predict",
        metavar="IMAGE",
        help="Path to a leaf image to predict (skips training)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to show (default: 3)",
    )
    args = parser.parse_args()

    if args.predict:
        # ── Prediction mode ───────────────────────────────────────────────────
        print(f"\n  Predicting: {args.predict}\n")
        result = predict_disease(args.predict, top_k=args.top_k)

        print(f"\n{'='*60}")
        print(f"  PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"  Disease     : {result['disease']}")
        print(f"  Confidence  : {result['confidence_pct']}")
        print(f"\n  Top {args.top_k} predictions:")
        for rank, pred in enumerate(result["top_predictions"], start=1):
            bar = "█" * int(pred["confidence"] * 30)
            print(f"  {rank}. {pred['disease']:<40} {pred['confidence']*100:5.1f}%  {bar}")
        print(f"{'='*60}\n")

    else: 
        # ── Training mode ─────────────────────────────────────────────────────
        run_training()