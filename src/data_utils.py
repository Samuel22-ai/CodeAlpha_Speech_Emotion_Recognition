import os
import joblib
import numpy as np
from typing import Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

def prepare_data(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_state: int = 42, 
    save_dir: str = "../models"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Prepares raw audio feature matrices and labels for Convolutional Neural Network (CNN) training.
    
    This function handles string-to-categorical label encoding, stratified train/test splitting,
    feature standardization (scaling) to prevent data leakage, and tensor reshaping to add 
    the necessary channel dimension for 2D convolutions. It also serializes the fitted 
    encoders and scalers to disk for future inference on raw data.

    Args:
        X (np.ndarray): The input feature matrix (e.g., extracted MFCCs) with shape (samples, height, width).
        y (np.ndarray): The 1D array of string labels corresponding to the audio samples.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.
        save_dir (str, optional): Directory path where the fitted StandardScaler and LabelEncoder will be saved. Defaults to "../models".

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: A tuple containing:
            - X_train_final (np.ndarray): Scaled and reshaped training features (samples, height, width, 1).
            - X_test_final (np.ndarray): Scaled and reshaped testing features (samples, height, width, 1).
            - y_train (np.ndarray): One-hot encoded training labels.
            - y_test (np.ndarray): One-hot encoded testing labels.
            - num_classes (int): The total number of unique target classes.

    Example:
        >>> import numpy as np
        >>> X = np.random.rand(1440, 20, 125)
        >>> y = np.array(['happy', 'sad', 'neutral'] * 480)
        >>> X_train, X_test, y_train, y_test, num_classes = prepare_data(X, y)
        >>> print(X_train.shape)
        (1152, 20, 125, 1)
    """
    
    # Ensure the models directory exists to save our deployment assets
    os.makedirs(save_dir, exist_ok=True)
    
    # ==========================================
    # 1. LABEL ENCODING (String -> Int -> One-Hot)
    # ==========================================
    # Initialize the LabelEncoder to convert string labels into integers.
    label_encoder = LabelEncoder()
    
    # fit_transform learns the mapping from string labels to integers and applies it.
    # It scans the array, discovers unique values, alphabetizes them, and replaces 
    # every string label with its corresponding new integer ID.
    y_encoded = label_encoder.fit_transform(y)
    
    # Convert integer-encoded labels into one-hot encoded vectors. 
    # It builds a matrix where each row corresponds to a sample and each column to a class (1 if belongs, 0 otherwise).
    # This prevents the model from assuming an ordinal relationship between classes.
    y_categorical = to_categorical(y_encoded)
    num_classes = y_categorical.shape[1]
    
    # SAVE THE ENCODER: Mandatory for Phase 4 (Inference). We must know what '0' or '1' means later.
    joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.joblib'))
    
    # ==========================================
    # 2. STRATIFIED TRAIN/TEST SPLIT
    # ==========================================
    # Implementing stratify=y_encoded guarantees the exact same proportion of emotions in both sets. 
    # Example for RAVDESS: If we have 100 samples of emotion A and 50 samples of emotion B, without 
    # stratification, we might end up with a biased training set. Stratification ensures both sets 
    # have the same proportion, helping the model learn all emotions effectively.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # ==========================================
    # 3. FEATURE SCALING (Preventing Data Leakage)
    # ==========================================
    scaler = StandardScaler()
    train_shape = X_train.shape
    test_shape = X_test.shape
    
    # Scikit-learn scalers expect 2D arrays. This is a three-chain command:
    # 1. reshape: flattens the 3D array into a 2D array.
    # 2. fit_transform / transform: standardizes features (removes mean, scales to unit variance).
    #    We only 'fit' on training data to prevent data leakage.
    # 3. reshape: converts the scaled 2D array back into its original 3D shape.
    X_train_scaled = scaler.fit_transform(X_train.reshape(train_shape[0], -1)).reshape(train_shape)
    X_test_scaled = scaler.transform(X_test.reshape(test_shape[0], -1)).reshape(test_shape)
    
    # SAVE THE SCALER: Mandatory for Phase 4. We must apply this exact mathematical scaling to incoming raw audio.
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    
    # ==========================================
    # 4. DIMENSIONALITY (Adding the Channel)
    # ==========================================
    # As only 4D tensors are accepted by CNNs, we need to add a channel dimension.
    # np.expand_dims adds a new axis at the end of the array (axis=-1).
    # Since we are working with mono audio, we add a channel dimension of size 1.
    # This results in a shape of (num_samples, height, width, 1).
    X_train_final = np.expand_dims(X_train_scaled, axis=-1)
    X_test_final = np.expand_dims(X_test_scaled, axis=-1)
    
    return X_train_final, X_test_final, y_train, y_test, num_classes