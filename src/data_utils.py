import os
import joblib
import numpy as np
from typing import Tuple, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

def prepare_data(
    X: np.ndarray, 
    y: np.ndarray, 
    actors: np.ndarray,
    save_dir: str = "../models"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Prepares raw audio feature matrices and labels for Convolutional Recurrent Neural Network (CRNN) training.
    
    This pipeline executes four critical preprocessing steps:
    1. Subject-Independent Splitting: Separates training and testing data by actor ID to prevent biometrics memorization.
    2. Label Encoding: Transforms string emotion labels into strictly one-hot encoded integer arrays.
    3. Feature Scaling: Standardizes MFCC features using training statistics to prevent data leakage.
    4. Dimensionality Expansion: Reshapes the 3D matrices into 4D tensors required by Keras 2D Convolutions.

    It also serializes the fitted `LabelEncoder` and `StandardScaler` to disk, which are absolutely 
    mandatory for maintaining consistency during the real-time inference/deployment phase.

    Args:
        X (np.ndarray): The input feature matrix (e.g., extracted MFCCs) with shape (samples, height, width).
        y (np.ndarray): The 1D array of string emotion labels corresponding to the audio samples.
        actors (np.ndarray): The 1D array of integer actor IDs corresponding to the audio samples.
        save_dir (str, optional): Directory path where the fitted StandardScaler and LabelEncoder 
                                  will be saved. Defaults to "../models".

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: A tuple containing:
            - X_train_final (np.ndarray): Scaled and reshaped training features (samples, height, width, 1).
            - X_test_final (np.ndarray): Scaled and reshaped testing features (samples, height, width, 1).
            - y_train (np.ndarray): One-hot encoded training labels.
            - y_test (np.ndarray): One-hot encoded testing labels.
            - num_classes (int): The total number of unique target emotion classes.

    Raises:
        ValueError: If the `train_mask` is empty (e.g., no actors <= 20 were found).

    Example:
        >>> import numpy as np
        >>> X = np.random.rand(1440, 60, 125)  # 60 = 20 MFCCs * 3 (base + delta + delta-delta)
        >>> y = np.array(['happy', 'sad', 'neutral'] * 480)
        >>> actors = np.array([1, 2, 21, 22] * 360)
        >>> X_train, X_test, y_train, y_test, num_classes = prepare_data(X, y, actors)
        >>> print(X_train.shape)
        (720, 60, 125, 1)
    """
    
    # Ensure the models directory exists to save our deployment assets (scalers/encoders)
    os.makedirs(save_dir, exist_ok=True)
    
    # ==========================================
    # 1. SUBJECT-INDEPENDENT TRAIN/TEST SPLIT
    # ==========================================
    # We use Actors 1-20 for training and completely unseen Actors 21-24 for testing.
    # Why? If we randomly split, the model might just memorize the specific acoustic properties
    # (timbre, pitch base) of a person's voice rather than the universal patterns of the emotion.
    # This subject-independent split guarantees the model generalizes to new, unseen speakers.
    print(f"Total samples before split: {len(actors)}")
    print(f"Unique actors found in dataset: {np.unique(actors)}")

    train_mask = actors <= 20
    test_mask = actors > 20

    # CRITICAL CHECK: Ensure our threshold actually captured training data
    if np.sum(train_mask) == 0:
        raise ValueError(f"train_mask is empty! Actors found were: {np.unique(actors)}. Adjust your mask threshold.")
    
    X_train, y_train_raw = X[train_mask], y[train_mask]
    X_test, y_test_raw = X[test_mask], y[test_mask]

    print(f"Post-split Train shape: {X_train.shape}")
    print(f"Post-split Test shape: {X_test.shape}")
    
    # ==========================================
    # 2. LABEL ENCODING (String -> Int -> One-Hot)
    # ==========================================
    # Initialize the LabelEncoder to mathematically represent our string labels.
    label_encoder = LabelEncoder()
    
    # We `.fit_transform()` ONLY on the training data to define our class vocabulary.
    # We then only `.transform()` the test data. This strictly prevents data leakage.
    y_train_encoded = label_encoder.fit_transform(y_train_raw)
    y_test_encoded = label_encoder.transform(y_test_raw)
    
    # Convert ordinal integer labels (0, 1, 2...) into one-hot encoded vectors ([1,0,0], [0,1,0]...).
    # Neural networks perform better on classification tasks with one-hot encoding because it 
    # removes any false mathematical proximity (e.g., class 1 is not "closer" to class 2 than class 5).
    y_train = to_categorical(y_train_encoded)
    y_test = to_categorical(y_test_encoded)
    
    # Dynamically extract the number of classes for the final Dense layer of the CRNN
    num_classes = y_train.shape[1]
    
    # SAVE THE ENCODER: Mandatory for Phase 4 (Inference). 
    # When our model outputs a prediction index (e.g., `3`), we need this to decode it back to 'sad'.
    joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.joblib'))
    
    # ==========================================
    # 3. FEATURE SCALING (Preventing Data Leakage)
    # ==========================================
    # StandardScaler normalizes features so they have a mean of 0 and a standard deviation of 1.
    # This dramatically speeds up gradient descent convergence in deep learning models.
    scaler = StandardScaler()
    train_shape = X_train.shape
    test_shape = X_test.shape
    
    # Scikit-learn scalers strictly expect 2D arrays (samples, features), but our data is 3D (samples, height, width).
    # 1. Reshape: Flatten the height and width (MFCCs and time steps) into a single long feature vector per sample.
    # 2. Fit/Transform: We FIT only on the training set. The test set is transformed using the training set's mean/variance.
    # 3. Reshape: Fold the flat array back into its original 3D spatial shape for convolutions.
    X_train_scaled = scaler.fit_transform(X_train.reshape(train_shape[0], -1)).reshape(train_shape)
    X_test_scaled = scaler.transform(X_test.reshape(test_shape[0], -1)).reshape(test_shape)
    
    # SAVE THE SCALER: Mandatory for Phase 4 (Inference). 
    # Incoming raw mic audio must be scaled using the exact same statistical distribution as the training data.
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    
    # ==========================================
    # 4. DIMENSIONALITY (Adding the Channel)
    # ==========================================
    # Keras Conv2D layers strictly expect 4D input tensors: (batch_size, height, width, channels).
    # Since our MFCC "image" is fundamentally 2D and represents a mono audio signal, it has 1 channel (like grayscale).
    # np.expand_dims adds this required dummy channel at the very end (`axis=-1`), assuming "channels_last" format.
    X_train_final = np.expand_dims(X_train_scaled, axis=-1)
    X_test_final = np.expand_dims(X_test_scaled, axis=-1)
    
    return X_train_final, X_test_final, y_train, y_test, num_classes