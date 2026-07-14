import os
import glob
import librosa
import numpy as np
from typing import Tuple

# Dictionary mapping from the RAVDESS emotion codes to their corresponding emotion labels.
EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_mfcc_features(audio_data: np.ndarray, sr: int, n_mfcc: int, max_pad_len: int) -> np.ndarray:
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) and their 1st/2nd order 
    derivatives (deltas) from an audio signal, and forces a uniform shape.

    This function extracts the base MFCCs, calculates the delta and delta-delta 
    features to capture dynamic spectral characteristics, concatenates them along 
    the feature axis, and finally pads or truncates the time-step axis to ensure 
    the output tensor has exactly `max_pad_len` columns.

    Args:
        audio_data (np.ndarray): The raw 1D audio time series array.
        sr (int): The sampling rate of the audio signal (in Hz).
        n_mfcc (int): The number of base MFCC features to extract.
        max_pad_len (int): The target length of the time-step axis (columns). 
                           Shorter arrays are padded with zeros; longer ones are truncated.

    Returns:
        np.ndarray: A 2D feature matrix of shape (n_mfcc * 3, max_pad_len) containing
                    the concatenated base MFCCs, deltas, and delta-deltas.
    """
    # Extract base MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    
    # Extract 1st and 2nd order deltas
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Concatenate features along the coefficient axis (axis=0)
    combined_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
    
    # Pad or truncate to ensure consistent time-step width (max_pad_len)
    if combined_features.shape[1] < max_pad_len:
        pad_width = max_pad_len - combined_features.shape[1]
        padded_mfccs = np.pad(combined_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        padded_mfccs = combined_features[:, :max_pad_len]
        
    return padded_mfccs

def extract_and_serialize_features(data_dir: str, save_dir: str, max_pad_len: int = 125, n_mfcc: int = 20, sr: int = 16000) -> None:
    """
    Scans a directory for RAVDESS .wav files, extracts features, applies data 
    augmentations, and serializes the resulting matrices to disk.

    The function reads the RAVDESS filenames to extract emotion labels and actor IDs.
    For all actors, the original audio is processed. For training set actors (IDs 1-20),
    data augmentation (additive white noise and pitch shifting) is applied to simulate 
    different environmental and vocal conditions to improve model robustness. Finally,
    the feature matrices (X), labels (y), and actor IDs are saved as `.npy` binaries.

    Args:
        data_dir (str): The root directory containing the RAVDESS dataset .wav files.
                        It will be searched recursively.
        save_dir (str): The destination directory where the serialized `.npy` files 
                        will be saved.
        max_pad_len (int, optional): The fixed time-step length for the output features. 
                                     Defaults to 125.
        n_mfcc (int, optional): The number of base MFCCs to extract. Defaults to 20.
        sr (int, optional): The target sampling rate for all loaded audio. Defaults to 16000 Hz.

    Returns:
        None: The function writes `X_features.npy`, `y_labels.npy`, and `actor_ids.npy` 
              directly to the specified `save_dir`.
    """
    file_pattern = os.path.join(data_dir, "**", "*.wav")
    all_audio_files = glob.glob(file_pattern, recursive=True)
    
    if not all_audio_files:
        print(f"Error: No .wav files found in {data_dir}")
        return

    print(f"Discovered {len(all_audio_files)} files. Starting extraction...")
    
    print("\n--- EXTRACTION CONFIGURATION ---")
    print(f"Sample Rate: {sr} Hz")
    print(f"Base MFCCs (n_mfcc): {n_mfcc}")
    print(f"Total Feature Height (Base + 2 Deltas): {n_mfcc * 3}")
    print(f"Max Pad Length (Time Steps): {max_pad_len}")
    print("Augmentations: Active (Original, White Noise, Pitch Shift)")
    print("--------------------------------\n")
    
    X_features, y_labels, actors = [], [], []
    
    for i, file_path in enumerate(all_audio_files):
        filename = os.path.basename(file_path)
        
        # Parse RAVDESS filename format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        parts = filename.split('-')
        if len(parts) < 7:
            continue # Skip files that don't match standard RAVDESS naming
            
        emotion_code = parts[2]
        emotion = EMOTION_MAP.get(emotion_code, 'unknown')
        actor_id = int(parts[6].split('.')[0])
        
        # Load raw audio (capped at 4 seconds)
        audio_data, _ = librosa.load(file_path, sr=sr, duration=4)
        
        # === 1. ORIGINAL AUDIO (All Actors) ===
        X_features.append(extract_mfcc_features(audio_data, sr, n_mfcc, max_pad_len))
        y_labels.append(emotion)
        actors.append(actor_id)
        
        # === AUGMENTATIONS (Only for Training Actors 1-20) ===
        # Actors 21-24 are typically held out for validation/testing.
        if actor_id <= 20:
            # === 2. ADDITIVE WHITE NOISE ===
            # Simulates a staticky microphone or background hum
            noise_amp = 0.005 * np.random.uniform() * np.amax(audio_data)
            noisy_audio = audio_data + noise_amp * np.random.normal(size=audio_data.shape[0])
            X_features.append(extract_mfcc_features(noisy_audio, sr, n_mfcc, max_pad_len))
            y_labels.append(emotion)
            actors.append(actor_id)
            
            # === 3. PITCH SHIFTING ===
            # Simulates different vocal cord depths (higher or lower pitch)
            pitch_steps = np.random.choice([-1, -0.5, 0.5, 1]) # Shift up or down within 1 semitone
            pitched_audio = librosa.effects.pitch_shift(y=audio_data, sr=sr, n_steps=pitch_steps)
            X_features.append(extract_mfcc_features(pitched_audio, sr, n_mfcc, max_pad_len))
            y_labels.append(emotion)
            actors.append(actor_id)
            
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(all_audio_files)} source files...")
       
    X = np.array(X_features)
    y = np.array(y_labels)
    actor_ids = np.array(actors)
    
    print("\n--- EXTRACTION COMPLETE ---")
    print(f"Total Augmented Files: {len(X)}")
    print(f"X Shape: {X.shape}")
    print(f"y Shape: {y.shape}")
    
    # Save the extracted features
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_features.npy"), X)
    np.save(os.path.join(save_dir, "y_labels.npy"), y)
    np.save(os.path.join(save_dir, "actor_ids.npy"), actor_ids)
    print(f"Saved to {save_dir}")