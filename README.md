# Speech Emotion Recognition (SER)

**CodeAlpha Machine Learning Internship - Task 2**

## Objective

To build a robust Machine Learning system capable of recognizing human emotions (e.g., happy, angry, sad) from raw speech audio.

Beyond standard classification, this project aims to explore **Edge AI optimizations**, taking a high-accuracy Python model and reducing its memory footprint via quantization for potential microcontroller deployment.

## Technical Architecture

* **Language:** Python 3.x
* **Core Libraries:** TensorFlow/Keras, Librosa (Audio Processing), NumPy, Pandas, Scikit-Learn
* **Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
* **Digital Signal Processing (DSP):** Extraction of Mel-Frequency Cepstral Coefficients (MFCCs) to convert raw audio waves into visual spectrogram data.

## Project Phases (In Progress)

1. **Data Ingestion & DSP:** Load `.wav` files and extract MFCC features.
2. **Model Training:** Design and train a Convolutional Neural Network (CNN) to classify acoustic features.
3. **Model Evaluation:** Generate confusion matrices and calculate Precision/Recall.
4. **The TinyML Optimization (Stretch Goal):** Apply TensorFlow Lite Post-Training Quantization (INT8) to shrink the model size while maintaining accuracy.

## Repository Structure

```text
├── data/               # Raw audio datasets (ignored in git)
├── notebooks/          # Jupyter notebooks for data exploration and DSP
├── src/                # Python scripts for training and evaluation
├── models/             # Saved TFLite and Keras models
├── README.md           # Project documentation
└── .gitignore          # Ignored files