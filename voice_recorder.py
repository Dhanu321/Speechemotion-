import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write

def record_voice():
    """
    This function records your voice and overwrites the existing audio file
    in the specified destination each time it's executed.
    """
    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording
    destination = "speech_emotion_recognition/recordings/livevoice.wav"

    print("Say something:")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write(destination, fs, myrecording)  # Overwrite the file
    print(f"Voice recording saved to {destination}.")
