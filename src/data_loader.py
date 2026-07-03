import os
import glob
import librosa
import numpy as np

#We used this dictionary as a mapping from the RAVDESS emotion codes to their corresponding emotion labels. This ensuring that the extracted features are correctly associated with their respective emotions for training and evaluation purposes. 
EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_and_serialize_features(data_dir, save_dir, max_pad_len=125, n_mfcc=20, sr=16000):
    """
    Scans a directory for RAVDESS .wav files, extracts Mel-Frequency Cepstral 
    Coefficients (MFCCs), standardizes their temporal length, and serializes 
    the resulting feature and label tensors to disk.

    This function is designed to handle hierarchical directory structures recursively.
    It expects filenames to follow the standard RAVDESS naming convention where 
    the emotion code is the third integer block separated by dashes (e.g., '03-01-05...').

    Args:
        data_dir (str): Absolute or relative path to the root directory containing 
            the raw audio .wav files.
        save_dir (str): Directory where the output .npy files will be saved. 
            Created automatically if it does not exist.
        max_pad_len (int, optional): The fixed number of temporal frames to standardise 
            the audio to. Shorter clips are zero-padded; longer clips are truncated. 
            Defaults to 125.
        n_mfcc (int, optional): The number of MFCC features to extract per frame. 
            Defaults to 20.
        sr (int, optional): The target sample rate to resample the audio to upon 
            loading. Defaults to 16000 Hz.

    Returns:
        None: The function does not return objects in memory. It saves two files 
        to `save_dir`: 'X_features.npy' and 'y_labels.npy'.

    Notes:
        - **Data Assumptions:** Currently optimised for studio-clean RAVDESS data.
        - **Future Deployment:** For noisy, real-world data (e.g., live microphone feed), 
          Voice Activity Detection (VAD) via `librosa.effects.trim` is highly recommended 
          prior to MFCC extraction to prevent modeling background noise.

    Example:
        >>> from src.data_loader import extract_and_serialize_features
        >>> extract_and_serialize_features(
        ...     data_dir='../data/raw', 
        ...     save_dir='../data',
        ...     max_pad_len=125,
        ...     n_mfcc=20
        ... )
        Discovered 1440 files. Starting extraction...
        --- EXTRACTION COMPLETE ---

    NOTE: For future real-world deployments (e.g., live microphone data), 
    silence trimming (Voice Activity Detection) is highly recommended before 
    MFCC extraction to prevent the model from learning ambient background noise. 
    See the commented-out trimming section inside the extraction loop.
    """
    
    file_pattern = os.path.join(data_dir, "**", "*.wav")
    all_audio_files = glob.glob(file_pattern, recursive=True)
    
    if not all_audio_files:
        print(f"Error: No .wav files found in {data_dir}")
        return

    print(f"Discovered {len(all_audio_files)} files. Starting extraction...")
    
    X_features, y_labels = [], []
    
    for i, file_path in enumerate(all_audio_files):
        # Extract the emotion code from the filename (e.g., '03' for happy)
        filename = os.path.basename(file_path)
        emotion_code = filename.split('-')[2]  # RAVDESS filenames have the emotion code in the 3rd segment which is the second index after splitting by '-'
        emotion = EMOTION_MAP[emotion_code] # Using the previously created emotion map dictionary to get the corresponding emotion label using the key-value relationship
        
        # Load audio
        audio_data, _ = librosa.load(file_path, sr=sr, duration=4)
        
        # --- FUTURE DEPLOYMENT OPTION: Silence Trimming ---
        # To enable Voice Activity Detection for noisy real-world data, 
        # uncomment the line below to trim decibels below the top 30dB threshold:
        # audio_data, _ = librosa.effects.trim(audio_data, top_db=30)
        # --------------------------------------------------
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        
        # Pad features
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            padded_mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            padded_mfccs = mfccs[:, :max_pad_len]
            
        X_features.append(padded_mfccs)
        y_labels.append(emotion)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(all_audio_files)} files...")
       
    # Convert to NumPy arrays
    X = np.array(X_features)
    y = np.array(y_labels)
    
    print("\n--- EXTRACTION COMPLETE ---")
    print(f"X Shape: {X.shape}")
    print(f"y Shape: {y.shape}")
    
    # Serialize arrays
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_features.npy"), X)
    np.save(os.path.join(save_dir, "y_labels.npy"), y)
    print(f"Saved to {save_dir}")