"""This module trains three neural network models on 
the dataset recordings and saves the X and y features."""

import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import ( # type: ignore
    Dense,
    Conv1D,
    Flatten,
    Dropout,
    Activation,
    MaxPooling1D,
    BatchNormalization,
    LSTM,
)
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create it if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def mlp_classifier(X, y):
    """Train an MLP classifier and save the results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100,),
        solver="adam",
        alpha=0.001,
        shuffle=True,
        verbose=True,
        momentum=0.8,
    )

    mlp_model.fit(X_train, y_train)

    mlp_pred = mlp_model.predict(X_test)
    mlp_accuracy = mlp_model.score(X_test, y_test)
    print("Accuracy: {:.2f}%".format(mlp_accuracy * 100))

    # Save classification report
    features_dir = os.path.abspath("speech_emotion_recognition/features")
    ensure_directory_exists(features_dir)

    mlp_clas_report = pd.DataFrame(
        classification_report(y_test, mlp_pred, output_dict=True)
    ).transpose()
    mlp_clas_report.to_csv(os.path.join(features_dir, "mlp_clas_report.csv"))
    print(classification_report(y_test, mlp_pred))


def lstm_model(X, y):
    """Train an LSTM model and save the results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_lstm = np.expand_dims(X_train, axis=2)
    X_test_lstm = np.expand_dims(X_test, axis=2)

    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(40, 1), return_sequences=True))
    lstm_model.add(LSTM(32))
    lstm_model.add(Dense(32, activation="relu"))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(8, activation="softmax"))

    lstm_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    lstm_model.summary()

    # Train model
    lstm_history = lstm_model.fit(X_train_lstm, y_train, batch_size=32, epochs=100)

    # Evaluate model on test set
    test_loss, test_acc = lstm_model.evaluate(X_test_lstm, y_test, verbose=2)
    print("\nTest accuracy:", test_acc)

    # Ensure directories exist
    images_dir = os.path.abspath("speech_emotion_recognition/images")
    ensure_directory_exists(images_dir)

    # Plot model loss
    plt.plot(lstm_history.history["loss"])
    plt.title("LSTM model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(os.path.join(images_dir, "lstm_loss.png"))
    plt.close()

    # Plot model accuracy
    plt.plot(lstm_history.history["accuracy"])
    plt.title("LSTM model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.savefig(os.path.join(images_dir, "lstm_accuracy.png"))
    plt.close()


def cnn_model(X, y):
    """Train a CNN model and save the results."""
    # Ensure directories exist
    images_dir = os.path.abspath("speech_emotion_recognition/images")
    ensure_directory_exists(images_dir)

    # One-hot encode the target labels
    num_classes = len(np.unique(y))
    y_one_hot = to_categorical(y, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    model = Sequential()
    model.add(Conv1D(16, 5, padding="same", input_shape=(40, 1)))
    model.add(Activation("relu"))
    model.add(Conv1D(8, 5, padding="same"))
    model.add(Activation("relu"))
    model.add(Conv1D(8, 5, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    cnn_history = model.fit(
        x_traincnn,
        y_train,
        batch_size=50,
        epochs=100,
        validation_data=(x_testcnn, y_test),
    )

    # Plot model loss
    plt.plot(cnn_history.history["loss"])
    plt.plot(cnn_history.history["val_loss"])
    plt.title("CNN model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    plt.savefig(os.path.join(images_dir, "cnn_loss.png"))
    plt.close()

    # Plot model accuracy
    plt.plot(cnn_history.history["accuracy"])
    plt.plot(cnn_history.history["val_accuracy"])
    plt.title("CNN model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    plt.savefig(os.path.join(images_dir, "cnn_accuracy.png"))
    plt.close()

    # Evaluate the model
    cnn_pred = np.argmax(model.predict(x_testcnn), axis=1)
    y_test_int = np.argmax(y_test, axis=1)

    matrix = confusion_matrix(y_test_int, cnn_pred)
    print(matrix)

    plt.figure(figsize=(12, 10))
    emotions = sorted(np.unique(y))  # Dynamically determine labels
    cm = pd.DataFrame(matrix)
    ax = sns.heatmap(
        matrix,
        linecolor="white",
        cmap="crest",
        linewidth=1,
        annot=True,
        fmt="",
        xticklabels=emotions,
        yticklabels=emotions,
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title("CNN Model Confusion Matrix", size=20)
    plt.xlabel("predicted emotion", size=14)
    plt.ylabel("actual emotion", size=14)
    plt.savefig(os.path.join(images_dir, "cnn_confusionmatrix.png"))
    plt.show()

    # Save classification report
    features_dir = os.path.abspath("speech_emotion_recognition/features")
    ensure_directory_exists(features_dir)

    clas_report = pd.DataFrame(
        classification_report(y_test_int, cnn_pred, output_dict=True)
    ).transpose()
    clas_report.to_csv(os.path.join(features_dir, "cnn_clas_report.csv"))
    print(classification_report(y_test_int, cnn_pred))

    # Save the model
    models_dir = os.path.abspath("speech_emotion_recognition/models")
    ensure_directory_exists(models_dir)

    model_path = os.path.join(models_dir, "cnn_model.h5")
    model.save(model_path)
    print("Saved trained model at %s " % model_path)


if __name__ == "__main__":
    print("Training started")

    # Check if files exist
    if not os.path.exists(r"E:/speech/speech_emotion_recognition/features/X.joblib"):
        print("X.joblib not found!")
    if not os.path.exists(r"E:/speech/speech_emotion_recognition/features/y.joblib"):
        print("y.joblib not found!")

    # Load the files
    X = joblib.load(r"E:/speech/speech_emotion_recognition/features/X.joblib")
    y = joblib.load(r"E:/speech/speech_emotion_recognition/features/y.joblib")

    # Debugging: Check the shape of X and y
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    cnn_model(X=X, y=y)
    print("Model finished.")