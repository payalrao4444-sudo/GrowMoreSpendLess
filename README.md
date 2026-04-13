# 🌿 AI Smart Kitchen Gardening System

A complete full-stack AI-powered urban kitchen garden assistant. This system integrates deep learning and dynamic environmental data to provide personalized crop recommendations, real-time disease detection, layout planning, and 100% organic gardening guidance.

## 🌟 Key Features

*   **🔐 Secure Authentication**: User accounts with session management and user activity/history tracking.
*   **🌤️ Dynamic Weather & AI Crop Recommendation**: Uses the OpenWeather API (or simulated fallback) combined with an AI scoring algorithm to recommend crops based on current temperature, humidity, and inferred seasons.
*   **🪴 Smart Container Detection**: Upload a photo of your planting container. The system uses a Convolutional Neural Network (CNN) to detect the size (small, medium, large) and provides customized crop suggestions that flourish in that space. Falls back to a robust heuristic image analysis if the CNN is unavailable.
*   **🔬 Plant Disease Diagnosis**: Deep learning classification of plant/leaf images to detect diseases (e.g., Early Blight, Powdery Mildew). Provides organic treatments, avoiding any chemical fertilizers. Uses heuristic fallback analysis based on color contrast, brightness, and standard deviations.
*   **🌱 Extensive Crop Database**: 40+ vegetables and herbs with detailed metadata including temp bounds, water requirements, harvest days, companion planting, and organic fertilizer schedules.
*   **🏡 Garden Layout Optimizer**: Submit your chosen crops, and the system evaluates companion planting compatibility, flagging any conflicting plants to maximize yield.
*   **♻️ 100% Organic Focus**: Emphasizes Vermicompost, Neem Oil, Wood Ash, and kitchen waste fertilizers. No chemical solutions are recommended.

## 📁 Project Structure

```text
c:\Users\DELL\OneDrive\Desktop\claud\
├── app.py                   # Main Flask application logic & API routes
├── complete_classifier.py   # Training script for the disease classifier
├── plant_disease.py         # Advanced disease detection logic
├── predict.py               # Inference script for diseases
├── train.py / train_model.py# Helper scripts for CNN training
├── requirements.txt         # Python dependencies
├── instance/
│   └── garden.db            # SQLite database (auto-created if Postgres is unavailable)
├── models/                  # Saved .keras/.h5 AI models
├── static/
│   └── uploads/             # Directory for user-uploaded images
└── templates/               # HTML templates (Dashboard, Login, Predict, etc.)
```

## 🛠️ Tech Stack

*   **Backend framework**: Python 3.8+ & Flask
*   **Database Engine**: PostgreSQL (Production) with SQLite fallback (Local)
*   **Machine Learning**: TensorFlow, Keras, NumPy, Pillow (PIL)
*   **Frontend UI**: HTML5, CSS3 (Glassmorphism design, modern animations), Vanilla JS
*   **External APIs**: OpenWeather API

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Make sure TensorFlow and Keras are installed successfully as they are required for the AI engines)*

### 4. Optional Configurations 

**Weather API:**
To enable real-time weather instead of simulated data, get a free API key from [openweathermap.org](https://openweathermap.org/api) and set it in your environment:
```bash
# Windows
set OPENWEATHER_API_KEY=your_api_key

# Mac/Linux
export OPENWEATHER_API_KEY="your_api_key"
```

**PostgreSQL (Optional):**
By default, the app builds a local SQLite database (`instance/garden.db`). To use PostgreSQL, set the `DATABASE_URL` environment variable:
```bash
export DATABASE_URL="postgresql://user:password@localhost/dbname"
```

### 5. Launch the Application
```bash
python app.py
```

### 6. Start Gardening!
Open your web browser and navigate to:
```text
http://localhost:5000
```

## 🔑 First Time Usage

1. Complete the sign-up process via the **"Create Account"** form on the homepage. Provide your location (e.g., `New Delhi`, `London`).
2. Explore the Dashboard: Check out your live weather telemetry and daily crop recommendations.
3. Access the AI center to analyze container sizes and leaf anomalies using image uploads.
4. Experiment with the Garden Layout planner to verify your companion farming.
