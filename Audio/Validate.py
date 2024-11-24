import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Load the saved model
model = load_model("drone_audio_classifier.h5")

# Parameters (these must match the parameters used during training)
sample_rate = 22050
max_duration = 5  # seconds
max_length = sample_rate * max_duration  # Maximum length in samples

# Label mapping (ensure this matches your training labels)
labels = ["unknown", "yes_drone"]  # Adjust based on your class labels

# Function to preprocess a single audio file
def preprocess_audio(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sample_rate)
        # Trim or pad the audio to match the required input length
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)))
        # Generate Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # Add channel dimension for CNN input
        mel_spectrogram = mel_spectrogram[..., np.newaxis]
        return mel_spectrogram
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

# Function to make a prediction
def predict_audio(file_path):
    # Preprocess the audio file
    mel_spectrogram = preprocess_audio(file_path)
    if mel_spectrogram is None:
        print("Error: Unable to process audio file.")
        return
    # Expand dimensions to match model input shape (1, height, width, channels)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    # Make prediction
    prediction = model.predict(mel_spectrogram)
    predicted_label = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display prediction as "Drone Present" or "Drone Not Detected"
    if predicted_label == "yes_drone":
        print(f"Drone Present (Confidence: {confidence:.2f}%)")
    else:
        print(f"Drone Not Detected (Confidence: {confidence:.2f}%)")

# Path to the audio file for validation
audio_file_path = r"C:\Users\heman\Desktop\Minor Proj\audio_minor\Binary_Drone_Audio\unknown\1-15689-A-42.wav" #ith your test audio file path

# Predict the class of the audio file
predict_audio(audio_file_path)
