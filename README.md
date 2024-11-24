# Detection-of-UAV-s-using-MULTI-MODEL-DATA-INPUTS



This project is a **multi-modal approach** to drone detection that leverages **YOLOv8** for image-based detection and a **deep learning-based audio classifier** for sound-based identification. By integrating these complementary techniques, the system can detect drones effectively using both visual and auditory cues.

## Features

- **YOLOv8 for Image Detection**: A cutting-edge object detection model identifies drones in images with high accuracy.
- **Audio Classification Using CNNs**: Mel spectrograms of audio data are analyzed to classify drone sounds using a convolutional neural network.
- **Robust Evaluation Metrics**: Includes confusion matrices, ROC curves, and precision-recall curves for a comprehensive assessment of model performance.
- **Interactive Visualizations**: Mel spectrogram plotting and training history visualizations for better interpretability.

## Dataset

### Image Detection
- **Input**: Drone images annotated for YOLOv8 training.
- **Output**: Bounding boxes around detected drones.

### Audio Detection
- **Input**: Binary classification dataset with drone and non-drone audio files.
- **Preprocessing**: 
  - Audio resampled at 22,050 Hz.
  - Converted to 5-second segments.
  - Represented as mel spectrograms for feature extraction.

## How It Works

1. **Audio Preprocessing**: 
   - Loads audio files and converts them into mel spectrograms.
   - Normalizes and encodes the data for model input.
2. **Model Architecture**:
   - **Conv2D Layers** for feature extraction.
   - **MaxPooling** and **Dropout Layers** for dimensionality reduction and regularization.
   - Dense layers with softmax for classification.
3. **Training**:
   - Data split into training and testing sets.
   - Model trained with categorical cross-entropy loss and Adam optimizer.
4. **Evaluation**:
   - Generates confusion matrices and AUC-ROC curves for validation.

## Results

- **Accuracy**: Achieves competitive accuracy for both audio and image-based drone detection.
- **Visual Outputs**:
  - Mel spectrogram samples.
  - Training and validation accuracy/loss graphs.
  - Confusion matrices and performance curves.



