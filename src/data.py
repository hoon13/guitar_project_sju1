import os
import random
import numpy as np
import librosa
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pedalboard import Pedalboard, Distortion, Convolution, PitchShift, LowShelfFilter, HighShelfFilter, PeakFilter

def rms_normalize(audio, target_rms=0.05):
    """
    RMS normalizes the input audio signal to a target RMS value.
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        return audio * (target_rms / current_rms)
    return audio

def generate_dataset(input_folder, output_folder, ir_file_path, total_samples=5000, sample_rate=16000, target_duration=5.0):
    """
    Integrated Data Factory:
    Loads DI -> random offset -> Input Normalization -> Effect Chain -> Output Normalization -> Mel Spectrogram -> Saved directly to PyTorch tensors.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    di_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    if not di_files:
        raise FileNotFoundError(f"No WAV files found in input folder: {input_folder}")
    if not os.path.exists(ir_file_path):
        raise FileNotFoundError(f"Cabinet IR file not found at: {ir_file_path}")
        
    print(f"[{len(di_files)} DI files found] Launching in-memory integrated data factory...")
    
    X_list = []
    y_list = []
    labels = []
    target_length = int(sample_rate * target_duration)
    
    for i in tqdm(range(total_samples), desc="Generating Data Factory Samples"):
        random_file = random.choice(di_files)
        file_path = os.path.join(input_folder, random_file)
        
        # 1. Random segment extraction (0.0 to 11.0 seconds offset)
        random_offset = random.uniform(0.0, 11.0)
        audio, _ = librosa.load(file_path, sr=sample_rate, mono=True, offset=random_offset, duration=target_duration)
        
        # Pad with zeros if short
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
            
        # 2. Input RMS normalization (remove volume bias)
        audio = rms_normalize(audio, target_rms=0.05)
        
        # Random parameters (Multi-task targets)
        gain_val = random.uniform(0.0, 1.0)
        bass_val = random.uniform(0.0, 1.0)
        mid_val = random.uniform(0.0, 1.0)
        treble_val = random.uniform(0.0, 1.0)
        pitch_shift = random.uniform(-7.0, 7.0)
        
        # 3. Effect chain (PitchShift -> Distortion -> 3-Band EQ -> Cabinet Convolution)
        board = Pedalboard([
            PitchShift(semitones=pitch_shift),
            Distortion(drive_db=gain_val * 55.0),
            LowShelfFilter(cutoff_frequency_hz=200.0, gain_db=(bass_val - 0.5) * 20.0),
            PeakFilter(cutoff_frequency_hz=1000.0, gain_db=(mid_val - 0.5) * 20.0, q=1.5),
            HighShelfFilter(cutoff_frequency_hz=3000.0, gain_db=(treble_val - 0.5) * 20.0),
            Convolution(ir_file_path, mix=1.0)
        ])
        
        effected_audio = board(audio, sample_rate)
        
        # 4. Output RMS normalization (remove volume bias)
        effected_audio = rms_normalize(effected_audio, target_rms=0.05)
        
        # 5. Mel Spectrogram conversion
        mel_spec = librosa.feature.melspectrogram(y=effected_audio, sr=sample_rate, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to float tensor and add channel dimension [1, H, W]
        mel_spec_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
        
        X_list.append(mel_spec_tensor)
        y_list.append([gain_val, bass_val, mid_val, treble_val])
        
        # Store metadata
        labels.append({
            'filename': f"sample_{i:04d}_MT_CAB.wav",
            'source_di': random_file,
            'offset': round(random_offset, 2),
            'pitch_shift': round(pitch_shift, 2),
            'gain_val': round(gain_val, 3),
            'bass_val': round(bass_val, 3),
            'mid_val': round(mid_val, 3),
            'treble_val': round(treble_val, 3)
        })
        
    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32)
    
    print(f"Data Generation Completed. X Shape: {X.shape}, y Shape: {y.shape}")
    
    # Train/Test Split
    indices = np.arange(total_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Save datasets
    torch.save((X_train, y_train), os.path.join(output_folder, 'train_data.pt'))
    torch.save((X_test, y_test), os.path.join(output_folder, 'test_data.pt'))
    print("Saved train_data.pt and test_data.pt successfully.")
    
    # Save metadata CSV
    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(output_folder, 'metadata.csv'), index=False)
    print("Metadata CSV saved successfully.")

def generate_real_dataset(real_sound_dir, output_folder, sample_rate=16000, target_duration=5.0, hop_duration=0.5):
    """
    Reads physically split real-world WAV files (from 'train' and 'val' subdirectories).
    Parses labels from filenames (e.g., '5,5,5,7.wav' -> [0.5, 0.5, 0.5, 0.7]).
    Slices them using a sliding window and saves directly as PyTorch tensors.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    train_dir = os.path.join(real_sound_dir, "train")
    val_dir = os.path.join(real_sound_dir, "val")
    
    target_length = int(sample_rate * target_duration)
    hop_length = int(sample_rate * hop_duration)
    
    def process_directory(directory_path, desc):
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return None, None
            
        wav_files = [f for f in os.listdir(directory_path) if f.endswith('.wav')]
        if not wav_files:
            print(f"No WAV files found in {directory_path}")
            return None, None
            
        X_list = []
        y_list = []
        
        for fname in tqdm(wav_files, desc=f"Processing {desc} real files"):
            # Parse label from name, e.g. "5,6,7,8.wav" -> [0.5, 0.6, 0.7, 0.8]
            label_parts = fname.replace('.wav', '').split(',')
            if len(label_parts) != 4:
                print(f"Skipping file with invalid format: {fname}")
                continue
            try:
                # 0-10 scale mapped to 0-1.0
                labels = [float(val) / 10.0 for val in label_parts]
            except ValueError:
                print(f"Skipping file with non-numeric labels: {fname}")
                continue
                
            fpath = os.path.join(directory_path, fname)
            audio, _ = librosa.load(fpath, sr=sample_rate, mono=True)
            
            # Check length constraint
            if len(audio) < target_length:
                # Pad with zeros if shorter than 5 seconds
                audio = np.pad(audio, (0, target_length - len(audio)))
                
            # Sliding window slicing
            for start_idx in range(0, len(audio) - target_length + 1, hop_length):
                segment = audio[start_idx : start_idx + target_length]
                segment = rms_normalize(segment, target_rms=0.05)
                
                # Mel Spectrogram
                mel_spec = librosa.feature.melspectrogram(y=segment, sr=sample_rate, n_mels=128, fmax=8000)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0) # [1, 128, W]
                
                X_list.append(mel_tensor)
                y_list.append(labels)
                
        if len(X_list) == 0:
            return None, None
            
        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.float32)
        return X, y

    print("--- Preparing Real Train Dataset ---")
    X_train, y_train = process_directory(train_dir, "Train")
    print("--- Preparing Real Validation Dataset ---")
    X_val, y_val = process_directory(val_dir, "Validation")
    
    if X_train is not None and y_train is not None:
        torch.save((X_train, y_train), os.path.join(output_folder, 'real_train_data.pt'))
        print(f"Saved real_train_data.pt. Shape: {X_train.shape}")
    if X_val is not None and y_val is not None:
        torch.save((X_val, y_val), os.path.join(output_folder, 'real_val_data.pt'))
        print(f"Saved real_val_data.pt. Shape: {X_val.shape}")

