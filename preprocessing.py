import os
import time
import joblib
import librosa
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create it if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def extract_file_info(data_path, save_dir):
    """Extract metadata from filenames and save as a CSV."""
    df = pd.DataFrame(columns=["file", "emotion"])

    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if not filename.endswith(".wav"):  # Ensure only .wav files are processed
                continue

            try:
                # Extract emotion from filename
                emotion = "unknown"  # Default emotion if not encoded in filename
                if "happy" in filename:
                    emotion = "happy"
                elif "angry" in filename:
                    emotion = "angry"
                elif "sad" in filename:
                    emotion = "sad"
                elif "neutral" in filename:
                    emotion = "neutral"

                # Use pd.concat instead of append
                df = pd.concat([df, pd.DataFrame({"file": [filename], "emotion": [emotion]})], ignore_index=True)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    ensure_directory_exists(save_dir)
    df.to_csv(os.path.join(save_dir, "df_file_features.csv"), index=False)
    print("File info extracted and saved.")


def extract_features(data_path, save_dir):
    """Extract MFCC features from audio files and save X and y."""
    feature_list = []
    label_list = []

    start_time = time.time()
    for dirpath, _, files in os.walk(data_path):
        for file in files:
            if not file.endswith(".wav"):  # Ensure only .wav files are processed
                continue

            file_path = os.path.join(dirpath, file)
            print(f"Processing file: {file_path}")  # Debugging statement

            try:
                y_lib, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
                mfccs = np.mean(
                    librosa.feature.mfcc(y=y_lib, sr=sample_rate, n_mfcc=40).T, axis=0
                )
                # Extract emotion label from filename
                if "happy" in file:
                    label = "happy"
                elif "angry" in file:
                    label = "angry"
                elif "sad" in file:
                    label = "sad"
                elif "neutral" in file:
                    label = "neutral"
                else:
                    label = "unknown"  # Default label if no emotion is encoded

                feature_list.append(mfccs)
                label_list.append(label)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    print("Data loaded in %s seconds." % (time.time() - start_time))

    if not feature_list:
        raise ValueError("No features were extracted. Check your dataset and file structure.")

    X = np.asarray(feature_list)
    y = np.asarray(label_list)
    print("Feature shapes:", X.shape, y.shape)

    ensure_directory_exists(save_dir)
    joblib.dump(X, os.path.join(save_dir, "X.joblib"))
    joblib.dump(y, os.path.join(save_dir, "y.joblib"))
    print("Features extracted and saved.")

    return X, y


def oversample(X, y, save_dir):
    """Perform oversampling to balance the dataset."""
    print("Original class distribution:", Counter(y))

    if len(set(y)) <= 1:
        print("Only one class detected. Skipping oversampling.")
        return X, y

    oversample = RandomOverSampler(sampling_strategy="not majority")
    X_over, y_over = oversample.fit_resample(X, y)

    print("Oversampled class distribution:", Counter(y_over))

    ensure_directory_exists(save_dir)
    joblib.dump(X_over, os.path.join(save_dir, "X_over.joblib"))
    joblib.dump(y_over, os.path.join(save_dir, "y_over.joblib"))
    print("Oversampled data saved.")

    return X_over, y_over


def train_model(X, y, save_dir):
    """Train a Random Forest model and save it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    ensure_directory_exists(save_dir)
    joblib.dump(model, os.path.join(save_dir, "model.pkl"))
    print("Model trained and saved.")

    return model


if __name__ == "__main__":
    # Define paths
    data_path = os.path.abspath("speech_emotion_recognition/recordings/")  # Path to audio files
    save_dir = os.path.abspath("speech_emotion_recognition/features/")  # Path to save features and model

    print(f"Data path: {data_path}")
    if os.path.exists(data_path):
        print(f"Files in data path: {os.listdir(data_path)}")
    else:
        print("The specified data path does not exist.")
        exit(1)  # Exit the script if the path does not exist

    # Step 1: Extract file info
    print("Extracting file info...")
    extract_file_info(data_path, save_dir)

    # Step 2: Extract audio features
    print("Extracting audio features...")
    X, y = extract_features(data_path, save_dir)

    # Step 3: Perform oversampling
    print("Performing oversampling...")
    X_over, y_over = oversample(X, y, save_dir)

    # Step 4: Train the model
    print("Training the model...")
    train_model(X_over, y_over, save_dir)

    print("Preprocessing and training completed.")