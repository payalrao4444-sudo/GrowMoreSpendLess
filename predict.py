import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ==========================================
# 1. Configuration Settings
# ==========================================
MODEL_PATH = os.path.join('models', 'disease_model.keras')
LABELS_PATH = os.path.join('models', 'disease_labels.json')
IMG_SIZE = (128, 128) # Must match the size used during training exactly

# ==========================================
# 2. Loading Model & Class Labels
# ==========================================
print("--- Initialization ---")
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not load model: {e}")
    sys.exit(1)

try:
    with open(LABELS_PATH, 'r') as f:
        # Load directly as string keys (e.g. "0": "Tomato_Early_blight")
        class_labels = json.load(f)
    print("Class labels loaded correctly.")
except Exception as e:
    print(f"ERROR: Could not load class labels {LABELS_PATH}: {e}")
    sys.exit(1)

# ==========================================
# 3. Prediction Function
# ==========================================
def predict_disease(img_path):
    if not os.path.exists(img_path):
         return {"disease": "undefined", "confidence": 0, "status": "error: file not found"}
    
    try:
        # 1. PREPROCESSING: Load the image and resize to match training size exactly
        img = image.load_img(img_path, target_size=IMG_SIZE)
        
        # Convert to numpy array & Add batch dimension -> Shape becomes (1, 128, 128, 3)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize pixel values (critical match to training)
        img_array = img_array / 255.0
        
        # 2. INFERENCE: Make the prediction
        predictions = model.predict(img_array, verbose=0)
        prob_array = predictions[0]
        
        predicted_class_index = int(np.argmax(prob_array))
        confidence_val = float(prob_array[predicted_class_index]) * 100
        idx_str = str(predicted_class_index)
        
        print("\n--- Model Prediction Debug ---")
        print(f"Full Prediction Array: {prob_array}")
        print(f"Max Index: {predicted_class_index}, Confidence: {confidence_val:.2f}%")
        
        # Detect poorly trained model
        if np.std(prob_array) < 0.05 or np.max(prob_array) < 0.3:
            print("⚠️ WARNING: Model not trained properly (predictions are almost identical)")

        print(f"Loaded Class Labels Keys: {list(class_labels.keys())}")
        
        # Robust dictionary fetching precisely requested by user
        if idx_str in class_labels:
            disease_name = class_labels[idx_str]
        else:
            try:
                disease_name = list(class_labels.values())[predicted_class_index]
            except Exception:
                disease_name = "Unknown Disease"
                
        disease_name = disease_name.replace('_', ' ')
        
        print(f"Mapped Disease Name Returned: {disease_name}")
        
        # 6. HANDLE WHOLE VEGETABLES & NON-LEAF WITH FALLBACK (< 30%)
        if confidence_val < 30.0:
            plant_type = disease_name.split()[0] if ' ' in disease_name else "Unknown Plant"
            fallback_disease = f"{plant_type}_General_Issue"
            print(f"Fallback triggered (< 30% conf). Reassigned to generic: {fallback_disease}")
            disease_name = fallback_disease
        
        return {
            "disease": disease_name,
            "confidence": round(confidence_val, 2),
            "status": "success"
        }
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return {"disease": "undefined", "confidence": 0, "status": "error during prediction"}

# ==========================================
# 4. Testing the Prediction Function
# ==========================================
if __name__ == "__main__":
    TEST_IMAGE_PATH = "test_image.jpg"
    print(f"\n--- Testing prediction on: {TEST_IMAGE_PATH} ---")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Warning: '{TEST_IMAGE_PATH}' doesn't exist.")
    else:
        result = predict_disease(TEST_IMAGE_PATH)
        print("\n" + "="*30)
        print("          RESULTS")
        print("="*30)
        print(json.dumps(result, indent=2))
        print("="*30)
