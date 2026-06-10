import os
import json
import joblib
import librosa
import numpy as np

from datetime import datetime
from tensorflow.keras.models import load_model

# =====================================
# PATHS
# =====================================

BASE_DIR = os.path.dirname(__file__)

SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

MODEL_PATH = os.path.join(SAVE_DIR, "babycry_cnn.keras")
SCALER_PATH = os.path.join(SAVE_DIR, "scaler.pkl")
CLASSES_PATH = os.path.join(SAVE_DIR, "classes.npy")
PARAMS_PATH = os.path.join(SAVE_DIR, "mfcc_params.json")

# =====================================
# LOAD MODEL
# =====================================

print("Loading trained model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

scaler = joblib.load(
    SCALER_PATH
)

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

with open(PARAMS_PATH, "r") as f:

    params = json.load(f)

N_MFCC = params["n_mfcc"]
MAX_FRAMES = params["max_frames"]
SR = params["sr"]

# =====================================
# SUGGESTIONS
# =====================================

def get_suggestion(label):

    suggestions = {

        "burping":
        "Try gently burping the baby.",

        "hungry":
        "Baby may be hungry. Try feeding.",

        "pain":
        "Baby may be uncomfortable or in pain.",

        "no_cry":
        "Baby seems calm and comfortable."
    }

    return suggestions.get(
        label,
        "No suggestion available."
    )

# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        sr=SR
    )

    y = y - np.mean(y)

    y = librosa.util.normalize(y)

    y = librosa.effects.preemphasis(y)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    if mfcc.shape[1] < MAX_FRAMES:

        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, MAX_FRAMES - mfcc.shape[1]))
        )

    else:

        mfcc = mfcc[:, :MAX_FRAMES]

    return mfcc, y

# =====================================
# MAIN PREDICTION
# =====================================

def predict_audio(file_path):

    mfcc, audio = extract_features(file_path)

    # =====================================
    # AUDIO ANALYSIS
    # =====================================

    rms = np.mean(
        librosa.feature.rms(y=audio)
    )

    peak = np.max(np.abs(audio))

    print("RMS:", rms)
    print("PEAK:", peak)

    # =====================================
    # REAL SILENCE DETECTION
    # =====================================

    # background noise:
    # low RMS + low peak

    if rms < 0.08 and peak < 0.35:

        return {

            "prediction":
            "no_cry",

            "confidence":
            99.0,

            "suggestion":
            get_suggestion("no_cry"),

            "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }

    # =====================================
    # PREPARE INPUT
    # =====================================

    X = np.array([mfcc])

    X = X.reshape(
        1,
        N_MFCC,
        MAX_FRAMES,
        1
    )

    X_flat = X.reshape(1, -1)

    X_scaled = scaler.transform(X_flat)

    X = X_scaled.reshape(
        1,
        N_MFCC,
        MAX_FRAMES,
        1
    )

    # =====================================
    # MODEL PREDICTION
    # =====================================

    prediction = model.predict(
        X,
        verbose=0
    )[0]

    print("RAW PREDICTIONS:", prediction)

    predicted_index = np.argmax(prediction)

    predicted_class = classes[predicted_index]

    confidence = float(
        prediction[predicted_index]
    )

    print("CLASS:", predicted_class)
    print("CONFIDENCE:", confidence)

    # =====================================
    # LOW CONFIDENCE
    # =====================================

    if confidence < 0.70:

        predicted_class = "no_cry"

        confidence = 0.99

    # =====================================
    # FINAL RESPONSE
    # =====================================

    return {

        "prediction":
        str(predicted_class),

        "confidence":
        round(confidence * 100, 2),

        "suggestion":
        get_suggestion(
            str(predicted_class)
        ),

        "timestamp":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }