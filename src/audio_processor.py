import os
import sys
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

if os.name == 'nt':
    ffmpeg_path = os.path.abspath("ffmpeg")
    if os.path.exists(ffmpeg_path) and ffmpeg_path not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_path + os.path.pathsep + os.environ["PATH"]

def separate_guitar(input_file_path, output_dir, shifts=10, overlap=0.5, apply_filter=True, low_hz=80.0, high_hz=7500.0):
    """
    Isolates the guitar stem from a mixture audio file using Demucs (htdemucs_6s).
    Applies custom-quality settings and an optional post-processing bandpass filter.
    Returns the path to the separated guitar WAV file.
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # We use the python interpreter from the current virtual environment to call demucs.separate
    python_exe = sys.executable
    cmd = [
        python_exe, "-m", "demucs.separate",
        "-n", "htdemucs_6s",
        "--two-stems=guitar",
        f"--shifts={shifts}",
        f"--overlap={overlap}",
        "-o", output_dir,
        input_file_path
    ]
    
    print(f"Running Demucs 6s separation (shifts={shifts}, overlap={overlap}) on {os.path.basename(input_file_path)}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Demucs separation failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        
    # Demucs output path structure: {output_dir}/htdemucs_6s/{track_name}/guitar.wav
    track_name = os.path.splitext(os.path.basename(input_file_path))[0]
    guitar_path = os.path.join(output_dir, "htdemucs_6s", track_name, "guitar.wav")
    
    if not os.path.exists(guitar_path):
        raise FileNotFoundError(f"Demucs finished but output file was not found at: {guitar_path}")
        
    # --- Post-processing: Butterworth Bandpass Filter ---
    if apply_filter:
        print(f"Post-processing: Applying bandpass filter ({low_hz}Hz-{high_hz}Hz) to isolate guitar range...")
        try:
            audio_data, fs = sf.read(guitar_path)
            
            # Design bandpass filter
            nyq = 0.5 * fs
            low = low_hz / nyq
            high = high_hz / nyq
            b, a = butter(4, [low, high], btype='band')
            
            # Apply filter to all channels
            if len(audio_data.shape) > 1:
                filtered_data = np.zeros_like(audio_data)
                for ch in range(audio_data.shape[1]):
                    filtered_data[:, ch] = lfilter(b, a, audio_data[:, ch])
            else:
                filtered_data = lfilter(b, a, audio_data)
                
            # Overwrite the file with filtered audio
            sf.write(guitar_path, filtered_data, fs)
            print("Bandpass filtering completed successfully.")
        except Exception as e:
            print(f"Warning: Bandpass filter failed (using raw separated audio instead): {str(e)}")
    else:
        print("Post-processing: Bandpass filter bypassed by user configuration.")
        
    print(f"Guitar stem isolated successfully: {guitar_path}")
    return guitar_path

