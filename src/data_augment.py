import random
import numpy as np

def apply_spec_augment(mfcc_array, freq_mask_param=3, time_mask_param=10):
    """Applies SpecAugment to a single MFCC matrix of shape (20, 125, 1)."""
    augmented_mfcc = mfcc_array.copy()
    
    # 1. Frequency Masking (Horizontal black bar)
    f = np.random.uniform(low=0.0, high=freq_mask_param)
    f = int(f)
    if f > 0:
        f0 = random.randint(0, augmented_mfcc.shape[0] - f)
        augmented_mfcc[f0:f0+f, :, 0] = 0 # Black out the frequencies
        
    # 2. Time Masking (Vertical black bar)
    t = np.random.uniform(low=0.0, high=time_mask_param)
    t = int(t)
    if t > 0:
        t0 = random.randint(0, augmented_mfcc.shape[1] - t)
        augmented_mfcc[:, t0:t0+t, 0] = 0 # Black out the time steps
        
    return augmented_mfcc

def augment_training_data(X_train, y_train):
    """Doubles the training data by appending a SpecAugmented copy of every sample."""
    print(f"Original training size: {X_train.shape[0]}")
    
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X_train)):
        # Keep the original
        augmented_X.append(X_train[i])
        augmented_y.append(y_train[i])
        
        # Create and keep a SpecAugmented copy
        spec_aug_version = apply_spec_augment(X_train[i])
        augmented_X.append(spec_aug_version)
        augmented_y.append(y_train[i])
        
    X_train_aug = np.array(augmented_X)
    y_train_aug = np.array(augmented_y)
    
    print(f"New augmented training size: {X_train_aug.shape[0]}")
    return X_train_aug, y_train_aug