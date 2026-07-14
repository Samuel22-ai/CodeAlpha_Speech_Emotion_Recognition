import random
import numpy as np
from typing import Tuple

def apply_spec_augment(mfcc_array: np.ndarray, freq_mask_param: int = 5, time_mask_param: int = 5) -> np.ndarray:
    """
    Applies SpecAugment (frequency and time masking) to a single MFCC feature matrix.

    SpecAugment is a state-of-the-art data augmentation technique for speech recognition.
    It works by randomly masking (setting to zero) continuous blocks of frequency channels 
    and time steps. This forces the neural network to learn robust, generalized features 
    rather than memorizing specific time-frequency patterns, thereby reducing overfitting.

    Args:
        mfcc_array (np.ndarray): A 3D numpy array representing the audio features for a single 
                                 sample, typically of shape (height, width, 1) after expanding dims.
        freq_mask_param (int, optional): The maximum width of the frequency mask (number of 
                                         consecutive frequency bands to mask). Defaults to 5.
        time_mask_param (int, optional): The maximum width of the time mask (number of 
                                         consecutive time steps to mask). Defaults to 5.

    Returns:
        np.ndarray: The augmented MFCC matrix with the same shape as the input, but with 
                    random horizontal and/or vertical black bars (zeros).
    """
    # Create a deep copy to ensure we don't modify the original array by reference
    augmented_mfcc = mfcc_array.copy()
    
    # ==========================================
    # 1. FREQUENCY MASKING (Horizontal Black Bar)
    # ==========================================
    # Pick a random mask width 'f' from [0, freq_mask_param]
    f = np.random.uniform(low=0.0, high=freq_mask_param)
    f = int(f)
    if f > 0:
        # Pick a random starting frequency band 'f0'
        # We ensure f0 + f doesn't exceed the total number of frequency bands (height)
        f0 = random.randint(0, augmented_mfcc.shape[0] - f)
        
        # Apply the mask: set the block [f0 : f0+f] across all time steps to zero
        augmented_mfcc[f0:f0+f, :, 0] = 0 
        
    # ==========================================
    # 2. TIME MASKING (Vertical Black Bar)
    # ==========================================
    # Pick a random mask width 't' from [0, time_mask_param]
    t = np.random.uniform(low=0.0, high=time_mask_param)
    t = int(t)
    if t > 0:
        # Pick a random starting time step 't0'
        # We ensure t0 + t doesn't exceed the total number of time steps (width)
        t0 = random.randint(0, augmented_mfcc.shape[1] - t)
        
        # Apply the mask: set the block [t0 : t0+t] across all frequency bands to zero
        augmented_mfcc[:, t0:t0+t, 0] = 0 
      
    return augmented_mfcc

def augment_training_data(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    freq_mask_param: int = 5, 
    time_mask_param: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expands the training dataset by appending a SpecAugmented copy of every single sample.

    This function effectively doubles the size of the training dataset. For every original 
    audio sample, it generates an augmented version using `apply_spec_augment` and pairs 
    it with the original label. This dramatically improves the CRNN model's ability to 
    generalize to noisy or distorted real-world audio inputs.

    Args:
        X_train (np.ndarray): The original training feature matrices of shape (samples, height, width, 1).
        y_train (np.ndarray): The original one-hot encoded training labels.
        freq_mask_param (int, optional): Maximum size for the frequency mask. Defaults to 5.
        time_mask_param (int, optional): Maximum size for the time mask. Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X_train_aug (np.ndarray): The new, doubled training feature matrix.
            - y_train_aug (np.ndarray): The new, doubled training label matrix.
    """
    print(f"Original training size: {X_train.shape[0]}")
    print(f"Augmentation Config: Max Freq Mask = {freq_mask_param}, Max Time Mask = {time_mask_param}")

    augmented_X = []
    augmented_y = []
    
    for i in range(len(X_train)):
        # 1. Keep the pristine, original sample
        augmented_X.append(X_train[i])
        augmented_y.append(y_train[i])
        
        # 2. Create and append a uniquely SpecAugmented copy of that sample
        spec_aug_version = apply_spec_augment(
            X_train[i], 
            freq_mask_param=freq_mask_param, 
            time_mask_param=time_mask_param
        )
        augmented_X.append(spec_aug_version)
        augmented_y.append(y_train[i])

        # Add a progress tracker that prints an update every 500 samples
        if (i + 1) % 500 == 0:
            print(f"Applied SpecAugment (Max Freq: {freq_mask_param}, Max Time: {time_mask_param}) to {i + 1}/{len(X_train)} samples...")
        
    # Convert lists back to NumPy arrays for Keras compatibility
    X_train_aug = np.array(augmented_X)
    y_train_aug = np.array(augmented_y)
    
    print(f"New augmented training size: {X_train_aug.shape[0]}")
    
    return X_train_aug, y_train_aug