import os
import streamlit as st
import soundfile as sf
import librosa
from predictions import make_predictions
from classification_app import load_data  # Assuming the classification logic is in 'classification_app.py'
import pickle
import numpy as np

# Load the pre-trained model for animal classification (pickled model)
try:
    model = pickle.load(open('model.pkl', 'rb'))  # Ensure model.pkl exists and is compatible
except Exception as e:
    st.error(f"Error loading classification model: {e}")
    st.stop()

st.title("Pet Emotion & Animal Classification")

# Ensure the recordings folder exists
os.makedirs("speech_emotion_recognition/recordings", exist_ok=True)

# Upload the audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "flac", "m4a"])

if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        raw_path = "speech_emotion_recognition/recordings/raw_uploaded_file.wav"
        with open(raw_path, "wb") as out_file:
            out_file.write(uploaded_file.getbuffer())

        # Convert to a consistent WAV format using librosa
        audio, sr = librosa.load(raw_path, sr=16000)
        processed_path = "speech_emotion_recognition/recordings/uploaded_audio.wav"
        sf.write(processed_path, audio, sr)

        # Play back the original uploaded file
        st.audio(uploaded_file, format="audio/wav")

        # Animal Classification Logic
        try:
            x_train, x_test, y_train, y_test = load_data(processed_path)
            
            # Check feature dimensions
            if x_test.shape[1] != model.n_features_in_:
                # Log the mismatch to the console instead of displaying it on the front-end
                print(f"[WARNING] Feature mismatch: Model expects {model.n_features_in_} features, but got {x_test.shape[1]}. Skipping animal classification.")
            else:
                animal_prediction = model.predict(x_test)[0]  # Get the animal prediction

                # Map prediction to animal label
                animal_map = ['Cat', 'Dog', 'Cow', 'Donkey', 'Monkey', 'Sheep']
                detected_animal = animal_map[animal_prediction]
                st.write(f"The detected animal is: **{detected_animal}**")
        except Exception as e:
            # Log the error for debugging but do not display it on the front-end
            print(f"[ERROR] Error during animal classification: {e}")

        # Emotion Detection Logic
        try:
            emotion = make_predictions(processed_path)
            st.write(f"The detected emotion of your pet is: **{emotion}**")
        except Exception as e:
            st.error(f"Error during emotion detection: {e}")

    except Exception as e:
        st.error(f"Error processing audio file: {e}")