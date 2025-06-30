import os
import numpy as np
import librosa
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_features_from_audio(file_path):
    """
    Extracts features (MFCCs, Chroma, Spectral Contrast, Tonnetz) from an audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        np.array: Extracted features as a NumPy array.
    """
    try:
        y_lib, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
        result = np.array([])

        # Extract MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=y_lib, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

        # Extract Chroma
        stft = np.abs(librosa.stft(y_lib))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
        # Extract Spectral Contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))

        # Extract Tonnetz
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y_lib), sr=sample_rate).T, axis=0)
        result = np.hstack((result, tonnetz))

        print(f"[INFO] Extracted features shape: {result.shape}")

        # Ensure the feature vector has exactly 53 features
        if result.shape[0] > 53:
            result = result[:53]
        elif result.shape[0] < 53:
            result = np.pad(result, (0, 53 - result.shape[0]))

        return result
    except Exception as e:
        print(f"[ERROR] Error processing file {file_path}: {e}")
        return None


def predict_emotion(file_path, model_path):
    """
    Predicts the emotion of an audio file using a pre-trained model.

    Args:
        file_path (str): Path to the audio file.
        model_path (str): Path to the pre-trained model.

    Returns:
        str: Predicted emotion.
    """
    # Extract features from the audio file
    features = extract_features_from_audio(file_path)
    if features is None:
        return "Error: Could not extract features from the audio file."

    # Load the pre-trained model
    if not os.path.exists(model_path):
        return "Error: Model file not found."

    model = joblib.load(model_path)

    # Ensure the feature dimensions match the model's input
    if features.shape[0] != model.n_features_in_:
        return f"Error: Feature mismatch. Model expects {model.n_features_in_} features, but got {features.shape[0]}."

    # Predict the emotion
    features = features.reshape(1, -1)  # Reshape for prediction
    predicted_emotion = model.predict(features)[0]

    # Debugging: Print prediction probabilities (if supported)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        print(f"[INFO] Prediction probabilities: {probabilities}")

    return predicted_emotion


if __name__ == "__main__":
    # Path to the recordings directory
    recordings_dir = r"E:/speech/speech_emotion_recognition/recordings"
    model_file_path = r"E:/speech/speech_emotion_recognition/model.pkl"  # Path to the pre-trained model

    # Check if the directory exists
    if not os.path.exists(recordings_dir):
        print(f"[ERROR] Directory '{recordings_dir}' does not exist.")
        exit()

    # List all audio files in the directory
    audio_files = [f for f in os.listdir(recordings_dir) if f.endswith(('.wav', '.mp3'))]
    if not audio_files:
        print(f"[WARNING] No audio files found in directory '{recordings_dir}'.")
        exit()
    # Process each audio file
    for audio_file in audio_files:
        audio_file_path = os.path.join(recordings_dir, audio_file)
        print(f"[INFO] Processing file: {audio_file_path}")

        # Predict the emotion
        emotion = predict_emotion(audio_file_path, model_file_path)
        print(f"[RESULT] Predicted Emotion for '{audio_file}': {emotion}")
