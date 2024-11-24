import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import sys

# Configure output encoding for Unicode
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Paths
dataset_path = "Binary_Drone_Audio"

# Parameters
sample_rate = 22050
max_duration = 5  # seconds
max_length = sample_rate * max_duration

# Function to plot Mel Spectrogram
def plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=sample_rate, cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Function to display confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Function to plot ROC Curve
def plot_roc_curve(y_true, y_pred, classes):
    plt.figure(figsize=(10, 6))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# Function to plot Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_pred, classes):
    plt.figure(figsize=(10, 6))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        plt.plot(recall, precision, label=f'{classes[i]}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

# Function to plot training history
def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Load audio data
def load_audio_data(dataset_path):
    labels = []
    features = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                try:
                    # Load and preprocess audio file
                    audio, sr = librosa.load(file_path, sr=sample_rate)
                    if np.sum(np.abs(audio)) < 1e-3:  # Skip silent audio files
                        continue
                    if len(audio) > max_length:
                        audio = audio[:max_length]
                    else:
                        audio = np.pad(audio, (0, max_length - len(audio)))
                    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
                    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                    features.append(mel_spectrogram)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return np.array(features), np.array(labels)

# Preprocess data
X, y = load_audio_data(dataset_path)
X = X[..., np.newaxis]  # Add channel dimension
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = tf.keras.utils.to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Plot training history
plot_training_history(history)

# Plot confusion matrix
plot_confusion_matrix(y_true_classes, y_pred_classes, label_encoder.classes_)

# Plot ROC Curve
plot_roc_curve(y_test, y_pred, label_encoder.classes_)

# Plot Precision-Recall Curve
plot_precision_recall_curve(y_test, y_pred, label_encoder.classes_)

# Save Model
model.save("drone_audio_classifier_new.keras")

# Visualize a sample Mel Spectrogram
def display_sample_spectrogram(dataset_path, num_samples=1):
    count = 0
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                try:
                    # Load and preprocess audio file
                    audio, sr = librosa.load(file_path, sr=sample_rate)
                    if len(audio) > max_length:
                        audio = audio[:max_length]
                    else:
                        audio = np.pad(audio, (0, max_length - len(audio)))
                    # Generate Mel Spectrogram
                    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
                    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=1.0)
                    # Plot and display spectrogram
                    plot_mel_spectrogram(mel_spectrogram_db, title=f"Mel Spectrogram of {file}")
                    count += 1
                    if count >= num_samples:
                        return
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

display_sample_spectrogram(dataset_path, num_samples=1)
