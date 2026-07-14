# Emotion Recognition from Speech (Edge AI & TinyML)

This repository contains an end-to-end Machine Learning pipeline for recognizing human emotions from speech audio, heavily optimized for Edge AI/TinyML deployment. The project transitions raw audio into a Spatially-Aligned Convolutional Recurrent Neural Network (CRNN), ultimately quantized to an 8-bit integer TensorFlow Lite model.

## Project Objective
* **Goal:** Recognize 8 human emotions (Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised) from raw audio waveforms.
* **Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song). Implemented a strict Subject-Independent Train/Test split (20 training actors, 4 unseen test actors) to mathematically prevent data leakage.
* **Target Metric:** > 60% Validation Accuracy on unseen actors.
* **Final Result:** 62.91% Validation Accuracy with a highly stable generalization gap, successfully quantized to an INT8 `.tflite` model for edge microcontrollers.

## Pipeline Evolution & Architecture Diagnostics
This project underwent a rigorous, 12-phase MLOps pipeline to overcome acoustic overlaps and tensor misalignments.

### Phases 1-3: The Baseline & Feature Extraction
* **Baseline CNN:** Extracted 20 MFCCs. Model hit a ceiling at ~53%, failing to distinguish high-arousal emotions (Fearful vs. Happy) due to a lack of temporal/velocity acoustic data.
* **Deep CNN + Deltas:** Expanded to 60 features (MFCCs + Deltas + Delta-Deltas) to capture acoustic velocity and acceleration. Increased model depth to break the plateau, hitting 58.33%. Replaced standard Flatten/Dense layers with Global Average Pooling (GAP) and Batch Normalization.
* **CRNN + Augmentation:** Integrated an LSTM (Long Short-Term Memory) layer to analyze temporal sequences. Applied audio augmentations (White Noise, Pitch Shift) and Tonal-Preserving SpecAugment (Freq Mask: 7, Time Mask: 15) to artificially expand the dataset without blinding the model to harmonic pitch.

### Phases 4-6: The Focal Loss Upgrade
* **Initial Validation Plateau:** The model stalled at ~57-62%.
* **Symmetric Dynamic Focal Loss:** Deployed a custom `@tf.keras.utils.register_keras_serializable()` class to securely manage a mutable Gamma tensor. This mathematically instructed the optimizer to drop gradients for well-classified examples, preventing "easy" examples from overwhelming the network, and applied targeted alpha weights to mitigate class imbalances.

### Phases 7-9: Spatial Tensor Alignment (The Breakthrough)
* **The "Silent Killer" Bug:** Discovered that flattening a `(3, 7, 128)` latent space into a 1D sequence directly via `Reshape((7, 384))` caused C-contiguous memory scrambling. The LSTM was receiving temporally mangled data, forcing it to memorize background noise (87% Train/57% Val).
* **The Fix:** Injected a `tf.keras.layers.Permute((2, 1, 3))` layer before reshaping to correctly transpose spatial dimensions to (Time, Frequency, Channels).
* **The Result:** Bridged the generalization gap. Reached 62.91% Validation Accuracy on unseen actors, demonstrating robust, un-overfitted temporal learning.

### Phases 10-11: Loss Landscape Diagnostics
* **The "Whack-a-Mole" Experiment:** Attempted to fix specific dataset overlaps (Sad vs. Disgust) using an Asymmetric Focal Loss penalty matrix (5.0x multiplier on specific false-positives).
* **Finding:** The asymmetric penalty forced the optimizer into an over-constrained landscape. The model overfit the training data (90.9% train) but validation stagnated at 62.91%. Concluded that the RAVDESS overlap between "Calm" and "Neutral" is a fundamental dataset flaw, not an architectural limitation. The Symmetric Loss model was selected as the superior generalizer.

### Phase 12: TinyML Shrinkage (Post-Training Quantization)
* Transitioned the 32-bit Float model to an 8-bit Integer model for microcontroller compatibility.
* Handled LSTM dynamic memory constraints (`tf.TensorListReserve` errors) by explicitly targeting `tf.lite.OpsSet.SELECT_TF_OPS` and disabling experimental tensor list lowering (`_experimental_lower_tensor_list_ops = False`).
* Shrank the model footprint by ~75% with negligible accuracy loss.

## Repository Architecture
```text
CodeAlpha_Emotion_Recognition/
├── data/                      # RAVDESS dataset (.npy and raw audio) 
|   ├── raw                 # Raw .wav files
|   ├── processed           # Processed .npy files
├── models/
│   ├── best_crnn_optimized_v9.keras # Phase 9 32-bit Float Model 
│   ├── emotion_model_int8.tflite # Phase 12 Quantized Edge Model 
│   ├── scaler.joblib             # Saved StandardScaler 
│   └── label_encoder.joblib      # Saved LabelEncoder 
├── notebooks/
│   ├── 01_Audio_Exploration.ipynb # EDA and initial data loader testing 
│   └── 02_CNN_Model_Building.ipynb # Core CRNN training, evaluation, and TFLite quantization 
├── src/
│   ├── data_loader.py            # Audio ingestion and MFCC/Delta extraction 
│   ├── data_utils.py             # Data handling and splitting utilities 
│   ├── data_augment.py           # SpecAugment and Waveform augmentation 
│   └── model_evaluation.py       # Confusion matrix and classification reports 
├── README.md
└── requirements.txt              # librosa, tensorflow, numpy, scikit-learn
```

## How to Reproduce

1. Clone this repository.
2. Download the RAVDESS dataset and place the raw audio files in the `data/` directory.
3. Run `notebooks/01_Audio_Exploration.ipynb` to ingest the audio, extract features, and serialize them into `.npy` arrays locally.
4. Run `notebooks/02_CNN_Model_Building.ipynb` to execute the full augmentation, training, evaluation, and TinyML quantization pipeline.

## Business Value & Edge Deployment

This project proves that deploying AI to the edge requires more than just compressing a model. By manually interrogating the confusion matrix, fixing spatial tensor geometry, and rejecting overfitted parameters, this pipeline produces a highly resilient audio classifier. This methodology is directly applicable to low-power IoT endpoints for automated customer service sentiment routing, remote telemetry, and voice-activated Edge AI.

