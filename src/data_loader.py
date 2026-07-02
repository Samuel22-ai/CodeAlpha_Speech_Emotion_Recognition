import os
import glob
import librosa
import numpy as np

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_and_serialize_features(data_dir, save_dir, max_pad_len=125, n_mfcc=20, sr=16000):
    """
    Scans a directory for RAVDESS .wav files, extracts padded MFCCs, 
    and saves the resulting X and y tensors to disk.
    
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
        # Parse label
        filename = os.path.basename(file_path)
        emotion_code = filename.split('-')[2]
        emotion = EMOTION_MAP[emotion_code]
        
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