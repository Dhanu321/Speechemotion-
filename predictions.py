import os
import numpy as np
import librosa
from tensorflow import keras


def make_predictions(file):
    """
    Predicts the emotion of the input audio file and returns a label.

    Parameters:
        file (str): Path to the audio file.

    Returns:
        str: Predicted emotion label.
    """
 
    # Load the models
    cnn_model_path = "E:/speech/speech_emotion_recognition/models/cnn_model.h5"
    # lstm_model_path = "C:/Users/syuva/OneDrive/Desktop/speech_emotion_recognition/models/lstm_model.h5"
        
    try:
        cnn_model = keras.models.load_model(cnn_model_path)
        # lstm_model = keras.models.load_model(lstm_model_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading models: {e}")

    # Load the audio file
    try:
        prediction_data, prediction_sr = librosa.load(
            file,
            res_type="kaiser_fast",
            duration=3,
            sr=22050,
            offset=0.5,
        )
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")

    # Extract MFCC features
    try:
        mfccs = np.mean(
            librosa.feature.mfcc(y=prediction_data, sr=prediction_sr, n_mfcc=40).T, axis=0
        )
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
    except Exception as e:
        raise ValueError(f"Error extracting MFCC features: {e}")

    # Get predictions from the LSTM model
    try:
        predictions = cnn_model.predict(x)
        predicted_class = predictions.argmax(axis=-1)[0]  # Get the predicted class index
    except Exception as e:
        raise RuntimeError(f"Error during model prediction: {e}")

    # Define emotions dictionary
    emotions_dict = {
        "0": "neutral",
        "1": "calm",
        "2": "happy",
        "3": "sad",
        "4": "angry",
        "5": "fearful",
        "6": "disgusted",
        "7": "surprised"
    }

    # Get the emotion label using the predicted class
    label = emotions_dict.get(str(predicted_class), "Unknown emotion")

    print(f"Predicted Emotion: {label}")

    return label



file_path = "E:/speech/speech_emotion_recognition/recordings/uploaded_audio.wav"

make_predictions(file_path)