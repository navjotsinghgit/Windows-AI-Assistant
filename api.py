from flask import Flask, request, jsonify
import pickle
import numpy as np
import librosa
import tempfile
import os

# ---------- LOAD PICKLE ----------
with open("voice_command_raw_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
label_encoder = data["label_encoder"]
config = data["config"]

SAMPLE_RATE = config["sample_rate"]
SAMPLES = config["samples"]
N_MFCC = config["n_mfcc"]

app = Flask(__name__)

# ---------- AUDIO FUNCTIONS ----------
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio)

    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    return audio, sr


def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return np.concatenate([mfcc_mean, mfcc_std])  # (80,)


# ---------- ROUTES ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Voice Command API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]

    # Save temp wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        audio, sr = preprocess_audio(tmp_path)
        features = extract_features(audio, sr)

        features = scaler.transform([features])
        probs = model.predict_proba(features)[0]

        idx = int(np.argmax(probs))
        prediction = label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])

        if confidence < 0.7:
            return jsonify({
                "prediction": "uncertain",
                "confidence": confidence
            })

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
