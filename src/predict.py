import os
import torch
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from .models import get_multitask_resnet18
from .data import rms_normalize

def predict_blind_test(test_audio_folder, model_path, output_image_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model checkpoint file not found.")
    if not os.path.exists(test_audio_folder):
        raise FileNotFoundError(f"Audio folder not found: {test_audio_folder}")
        
    model = get_multitask_resnet18(num_outputs=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    audio_files = [f for f in os.listdir(test_audio_folder) if f.endswith('.wav')]
    print(f"Analyzing {len(audio_files)} hardware audio files...")
    
    results = []
    sample_rate = 16000
    target_length = int(sample_rate * 5.0)
    
    with torch.no_grad():
        for filename in audio_files:
            file_path = os.path.join(test_audio_folder, filename)
            
            # Load 5.0 seconds of audio
            audio, _ = librosa.load(file_path, sr=sample_rate, mono=True, duration=5.0)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
                
            # RMS Normalize to match training data prep (remove volume bias)
            audio = rms_normalize(audio, target_rms=0.05)
            
            # Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Form tensor [1, 1, H, W]
            input_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict
            output = model(input_tensor)
            scores = output.cpu().numpy()[0]
            scores = np.clip(scores * 100, 0, 100)  # scale to 0-100
            
            results.append({
                '파일명': filename,
                'Gain Score': round(scores[0], 1),
                'Bass Score': round(scores[1], 1),
                'Mid Score': round(scores[2], 1),
                'Treble Score': round(scores[3], 1)
            })
            
    if results:
        res_df = pd.DataFrame(results)
        # Sort by Gain Score
        res_df = res_df.sort_values(by='Gain Score').reset_index(drop=True)
        
        print("\n--- AI Blind Test Results (Sorted by Gain Score) ---")
        print(res_df.to_string(index=False))
        
        # Plot Grouped Bar Chart (max 15 files for clean display, or sample them)
        plot_df = res_df.head(15)  # limit display to first 15 for readability
        
        x = np.arange(len(plot_df))
        width = 0.2
        
        plt.figure(figsize=(14, 7))
        plt.bar(x - 1.5*width, plot_df['Gain Score'], width, label='Gain', color='crimson', edgecolor='black')
        plt.bar(x - 0.5*width, plot_df['Bass Score'], width, label='Bass', color='royalblue', edgecolor='black')
        plt.bar(x + 0.5*width, plot_df['Mid Score'], width, label='Mid', color='forestgreen', edgecolor='black')
        plt.bar(x + 1.5*width, plot_df['Treble Score'], width, label='Treble', color='goldenrod', edgecolor='black')
        
        plt.title('AI Blind Test: Multi-Task Tone Analysis (First 15 Files)', fontsize=16, fontweight='bold')
        plt.xlabel('Audio Files', fontsize=12)
        plt.ylabel('Score (0 to 100)', fontsize=12)
        plt.xticks(x, plot_df['파일명'], rotation=45, ha='right')
        plt.ylim(0, 110)
        plt.legend(loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        if output_image_path:
            plt.savefig(output_image_path, dpi=150)
            print(f"\nGrouped bar chart saved to: {output_image_path}")

    else:
        print("Warning: No audio files found to process.")
