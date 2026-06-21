import os
import sys

# Add local FFmpeg binaries and shared DLLs directory to system PATH
# This is required for torchaudio / torchcodec / demucs to decode audio on Windows.
if os.name == 'nt':
    ffmpeg_path = os.path.abspath("ffmpeg")
    if os.path.exists(ffmpeg_path) and ffmpeg_path not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_path + os.path.pathsep + os.environ["PATH"]

import tempfile
import torch
import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt

# Ensure src is importable
sys.path.append(os.path.abspath('.'))

import importlib
import src.models
import src.data
import src.audio_processor

importlib.reload(src.models)
importlib.reload(src.data)
importlib.reload(src.audio_processor)

from src.models import get_multitask_resnet18
from src.data import rms_normalize
from src.audio_processor import separate_guitar

# Page Configuration for Premium Look
st.set_page_config(
    page_title="Multi-Task Tone Analyzer & Coach",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Premium Styling (Vibrant Colors, Dark Mode, Glassmorphism)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #08090d;
        color: #e2e8f0;
    }
    
    .stApp {
        background-color: #08090d;
    }
    
    /* Title */
    h1 {
        background: linear-gradient(90deg, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.6rem !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    /* Flat Cards */
    .card {
        background: #111218;
        border: 1px solid #1e2029;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    
    /* Clean Primary Button */
    .stButton > button {
        background: linear-gradient(90deg, #475569, #334155) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.65rem 2rem !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: background 0.2s ease !important;
        cursor: pointer !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #334155, #1e293b) !important;
        color: #ffffff !important;
    }
    
    /* Similarity Score Box */
    .similarity-container {
        text-align: center;
        background: #111218;
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.75rem 0;
        box-shadow: 0 4px 20px rgba(71, 85, 105, 0.1);
    }
    .similarity-title {
        font-size: 0.85rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }
    .similarity-score {
        font-size: 3.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    
    /* Simple Parameter Box */
    .param-box {
        background: #15161f;
        border: 1px solid #1e2029;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .param-box:hover {
        transform: translateY(-2px);
    }
    .param-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .param-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
    }
    .param-diff {
        font-size: 0.9rem;
        font-weight: bold;
        border-top: 1px solid #1e2029;
        padding-top: 0.5rem;
        margin-top: 0.5rem;
    }
    
    /* Clean Coaching Box */
    .coach-box {
        background: #111218;
        border: 1px solid #1e2029;
        border-left: 4px solid #475569;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        color: #f8fafc;
    }
    .coach-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #cbd5e1;
        margin-bottom: 0.85rem;
        letter-spacing: 0.5px;
    }
    .coach-item {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding-left: 0.5rem;
        color: #cbd5e1;
    }
    .coach-item-perfect {
        font-size: 1.2rem;
        font-weight: bold;
        color: #cbd5e1;
        text-align: center;
        padding: 0.5rem;
    }
    
    /* Contrast improvements for default Streamlit widgets */
    .stSlider label, .stMarkdown p, .stSubheader, .stAlert p, div[data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    
    /* Hide Streamlit default UI elements for a premium desktop app feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>


""", unsafe_allow_html=True)


# Helper function to predict tone parameters using Sliding Window & Noise Gate
def predict_tone(audio_path, model, device):
    sample_rate = 16000
    target_duration = 5.0
    target_length = int(sample_rate * target_duration)
    hop_duration = 1.0
    hop_length = int(sample_rate * hop_duration)
    rms_threshold = 0.005  # Noise Gate: skip frames with volume below this threshold
    
    # Load full audio
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio_len = len(audio)
    
    # List to collect prediction scores for each valid window
    window_scores = []
    
    if audio_len < target_length:
        # If segment is too short, pad with zeros and run once
        padded_audio = np.pad(audio, (0, target_length - audio_len))
        # Remove volume bias
        padded_audio = rms_normalize(padded_audio, target_rms=0.05)
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=padded_audio, sr=sample_rate, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        input_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            scores = output.cpu().numpy()[0]
            scores = np.clip(scores * 100, 0, 100)
        return scores
        
    # Sliding window inference
    for start_idx in range(0, audio_len - target_length + 1, hop_length):
        window = audio[start_idx : start_idx + target_length]
        
        # Calculate RMS of this window (Noise Gate check)
        window_rms = np.sqrt(np.mean(window ** 2))
        if window_rms < rms_threshold:
            # Skip silent parts to prevent diluting the tone signature
            continue
            
        # Normalize volume of active window
        normalized_window = rms_normalize(window, target_rms=0.05)
        
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=normalized_window, sr=sample_rate, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        input_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            scores = output.cpu().numpy()[0]
            scores = np.clip(scores * 100, 0, 100)
            window_scores.append(scores)
            
    # Fallback: if all windows were skipped because they were too quiet, predict on first window
    if not window_scores:
        first_window = audio[0:target_length]
        first_window = rms_normalize(first_window, target_rms=0.05)
        mel_spec = librosa.feature.melspectrogram(y=first_window, sr=sample_rate, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        input_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            scores = output.cpu().numpy()[0]
            scores = np.clip(scores * 100, 0, 100)
        return scores
        
    # Return average scores across all active playing windows
    avg_scores = np.mean(window_scores, axis=0)
    return avg_scores


# Helper function to trim audio and save as a temporary WAV file
def trim_audio(input_path, start_time, end_time):
    duration = end_time - start_time
    # Load segment at original sample rate
    y, sr = librosa.load(input_path, sr=None, offset=start_time, duration=duration)
    
    # Save to a temporary WAV file
    import soundfile as sf
    tmp_trimmed = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_trimmed_path = tmp_trimmed.name
    tmp_trimmed.close() # Close file handle for Windows compatibility
    sf.write(tmp_trimmed_path, y, sr)
    return tmp_trimmed_path

# Title
st.markdown("<h1>지능형 기타 톤 코칭 AI 시스템</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #a0aec0; margin-bottom: 3rem;'>Reference 음원에서 기타 트랙을 분리하여 분석하고, 최적의 이펙터 노브 설정 가이드를 제공합니다.</p>", unsafe_allow_html=True)

# Check model checkpoint
model_path = os.path.abspath("best_multi_task_resnet_model.pth")
if not os.path.exists(model_path):
    st.error("오류: 학습된 AI 모델 파일(best_multi_task_resnet_model.pth)을 찾을 수 없습니다. 먼저 Step 2 학습을 완료해 주세요.")
else:
    # Load Model once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_multitask_resnet18(num_outputs=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Layout - Uploaders
    col1, col2 = st.columns(2)
    
    ref_start, ref_end = 0.0, 5.0
    user_start, user_end = 0.0, 5.0
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("레퍼런스 음원 (Reference Song)")
        ref_file = st.file_uploader("닮고 싶은 기타 톤이 포함된 오디오 파일(.wav, .mp3)을 업로드하세요", type=['wav', 'mp3'], key="ref")
        
        if ref_file:
            ref_ext = os.path.splitext(ref_file.name)[1]
            if "ref_temp_path" not in st.session_state or st.session_state.get("ref_file_name") != ref_file.name:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ref_ext) as tmp:
                    tmp.write(ref_file.getvalue())
                    st.session_state.ref_temp_path = tmp.name
                    st.session_state.ref_file_name = ref_file.name
                    st.session_state.ref_duration = librosa.get_duration(path=tmp.name)
            
            ref_duration = st.session_state.ref_duration
            st.info(f"업로드 완료: {ref_file.name} (총 길이: {ref_duration:.1f}초)")
            
            ref_start, ref_end = st.slider(
                "기타 솔로 구간 선택 (초)",
                min_value=0.0,
                max_value=float(ref_duration),
                value=(0.0, min(5.0, float(ref_duration))),
                step=0.1,
                key="ref_slider"
            )
            st.write(f"선택된 구간: **{ref_start:.1f}초 ~ {ref_end:.1f}초** (길이: **{ref_end - ref_start:.1f}초**)")
            if ref_end - ref_start < 5.0:
                st.warning("정확한 AI 톤 분석을 위해 가급적 5초 이상의 구간을 선택해 주세요.")
            
            try:
                ref_segment, ref_sr = librosa.load(st.session_state.ref_temp_path, sr=None, offset=ref_start, duration=ref_end - ref_start)
                st.audio(ref_segment, sample_rate=ref_sr)
            except Exception as e:
                st.error(f"오디오 미리보기 재생 중 오류 발생: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("내 연주 녹음 (My Recording)")
        user_file = st.file_uploader("현재 나의 이펙터 설정 상태의 기타 단독 녹음 파일(.wav, .mp3)을 업로드하세요", type=['wav', 'mp3'], key="user")
        
        if user_file:
            user_ext = os.path.splitext(user_file.name)[1]
            if "user_temp_path" not in st.session_state or st.session_state.get("user_file_name") != user_file.name:
                with tempfile.NamedTemporaryFile(delete=False, suffix=user_ext) as tmp:
                    tmp.write(user_file.getvalue())
                    st.session_state.user_temp_path = tmp.name
                    st.session_state.user_file_name = user_file.name
                    st.session_state.user_duration = librosa.get_duration(path=tmp.name)
            
            user_duration = st.session_state.user_duration
            st.info(f"업로드 완료: {user_file.name} (총 길이: {user_duration:.1f}초)")
            
            user_start, user_end = st.slider(
                "분석할 연주 구간 선택 (초)",
                min_value=0.0,
                max_value=float(user_duration),
                value=(0.0, min(5.0, float(user_duration))),
                step=0.1,
                key="user_slider"
            )
            st.write(f"선택된 구간: **{user_start:.1f}초 ~ {user_end:.1f}초** (길이: **{user_end - user_start:.1f}초**)")
            if user_end - user_start < 5.0:
                st.warning("정확한 AI 톤 분석을 위해 가급적 5초 이상의 구간을 선택해 주세요.")
            
            try:
                user_segment, user_sr = librosa.load(st.session_state.user_temp_path, sr=None, offset=user_start, duration=user_end - user_start)
                st.audio(user_segment, sample_rate=user_sr)
            except Exception as e:
                st.error(f"오디오 미리보기 재생 중 오류 발생: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Analyze
    if ref_file and user_file:
        # Calculate estimated time dynamically:
        # Demucs with shifts=4 on CPU takes about 1.5s per second of padded audio.
        est_time = int((ref_end - ref_start + 15.0) * 1.5)
        st.markdown(f"<p style='text-align: center; color: #a0aec0; margin-bottom: 0.5rem;'><b>예상 분석 소요 시간</b>: 약 <b>{est_time}초</b> (연산 정밀도 Shifts=4 적용)</p>", unsafe_allow_html=True)
        
        if st.button("지능형 톤 매칭 분석 시작", use_container_width=True):
            ref_path = st.session_state.ref_temp_path
            user_path = st.session_state.user_temp_path
            
            shifts = 4
            overlap = 0.5
            apply_filter = False  # Bypassed filter to prevent Bass prediction mismatch
            low_hz = 80.0
            high_hz = 10000.0
            
            trimmed_ref_context_path = None
            trimmed_user_path = None
            ref_guitar_path = None
            
            # Initialize progress bar
            progress_bar = st.progress(0, text="분석 시작 준비 중...")
            
            try:
                # 1. Trim audio files (using wider context for reference to prevent boundary artifacts in Demucs)
                progress_bar.progress(10, text="10% - 선택한 오디오 솔로 구간 추출 중...")
                pad = 7.5  # 7.5 seconds padding on both sides to give 15+ seconds context
                demucs_start = max(0.0, ref_start - pad)
                demucs_end = min(st.session_state.ref_duration, ref_end + pad)
                
                trimmed_ref_context_path = trim_audio(ref_path, demucs_start, demucs_end)
                trimmed_user_path = trim_audio(user_path, user_start, user_end)
                
                # 2. Demucs Guitar track separation for Reference Song (uses padded context, much cleaner!)
                progress_bar.progress(30, text="30% - Demucs AI 모델을 사용하여 참조 음원으로부터 기타 파트 분리 중...")
                output_dir = os.path.abspath("dataset_final/separated")
                ref_guitar_context_path = separate_guitar(
                    trimmed_ref_context_path, 
                    output_dir,
                    shifts=shifts,
                    overlap=overlap,
                    apply_filter=apply_filter,
                    low_hz=low_hz,
                    high_hz=high_hz
                )
                
                # Slice the separated guitar back to the exact user selection
                relative_offset = ref_start - demucs_start
                ref_guitar_path = trim_audio(ref_guitar_context_path, relative_offset, relative_offset + (ref_end - ref_start))
                
                # 3. Analyze separated Reference & User direct recording
                progress_bar.progress(75, text="75% - AI 모델이 각 기타 트랙의 톤 질감 및 밸런스 분석 중...")
                ref_scores = predict_tone(ref_guitar_path, model, device)
                user_scores = predict_tone(trimmed_user_path, model, device)
                
                progress_bar.progress(95, text="95% - 결과 데이터 분석 및 시각화 준비 중...")
                
                # Calculate tone similarity score (100 - average error)
                mean_error = np.mean(np.abs(ref_scores - user_scores))
                similarity = max(0.0, 100.0 - mean_error)
                
                progress_bar.progress(100, text="100% - 분석 완료!")
                import time
                time.sleep(0.5)
                progress_bar.empty()
                
                # 1. Similarity Score Dashboard
                st.markdown(f"""
                    <div class='similarity-container'>
                        <div class='similarity-title'>Tone Matching Accuracy</div>
                        <div class='similarity-score'>{similarity:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # 5. Coaching guide based on Delta (Moved right below Similarity Dashboard)
                st.markdown("<div class='coach-box'>", unsafe_allow_html=True)
                st.markdown("<div class='coach-title'>AI 지능형 톤 코칭 가이드</div>", unsafe_allow_html=True)
                
                features = ['Gain', 'Bass', 'Mid', 'Treble']
                coaching_items = []
                for i, feat in enumerate(features):
                    delta = ref_scores[i] - user_scores[i]
                    abs_delta = abs(delta)
                    
                    if abs_delta >= 7.5:
                        action = "올리세요" if delta > 0 else "낮추세요"
                        coaching_items.append(f"**{feat}**을(를) **{abs_delta:.0f}%** {action}")
                
                if coaching_items:
                    for item in coaching_items:
                        st.markdown(f"<div class='coach-item'>{item}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='coach-item-perfect'>완벽합니다! 현재 사용자의 톤이 Reference 톤과 아주 잘 매칭되어 있습니다. 추가 조정이 필요 없습니다.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                    
                # 2. Parameter Cards
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("톤 분석 결과")
                
                acol1, acol2, acol3, acol4 = st.columns(4)
                features = ['Gain', 'Bass', 'Mid', 'Treble']
                colors = ['#ff4b2b', '#1e90ff', '#2ed573', '#ffa500']
                
                for i, feat in enumerate(features):
                    with [acol1, acol2, acol3, acol4][i]:
                        delta = ref_scores[i] - user_scores[i]
                        sign = "+" if delta >= 0 else ""
                        delta_color = "#8b5cf6" if abs(delta) < 7.5 else ("#ff4b2b" if delta > 0 else "#ffa500")
                        
                        st.markdown(f"""
                            <div class='param-box' style='border-top: 3px solid {colors[i]};'>
                                <div class='param-title' style='color: {colors[i]};'>{feat}</div>
                                <div class='param-row'>
                                    <span>Reference</span>
                                    <span style='font-weight: bold;'>{ref_scores[i]:.1f}%</span>
                                </div>
                                <div class='param-row'>
                                    <span>My Tone</span>
                                    <span style='font-weight: bold;'>{user_scores[i]:.1f}%</span>
                                </div>
                                <div class='param-diff' style='color: {delta_color};'>
                                    Diff: {sign}{delta:.1f}%
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 3. Audio player for isolated Reference Guitar track
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("추출된 레퍼런스 기타 소리 듣기 (Isolated Guitar Track)")
                try:
                    st.audio(ref_guitar_path, format="audio/wav")
                except Exception as e:
                    st.error(f"추출된 기타 소리를 로드하는 중 오류 발생: {str(e)}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 4. Matplotlib Visual chart
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("톤 밸런스 비교 그래프")
                
                fig, ax = plt.subplots(figsize=(5.0, 1.8))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                x = np.arange(len(features))
                width = 0.32
                
                ax.bar(x - width/2, ref_scores, width, label='Reference (Guitar separated)', color='#7f9ab5', alpha=0.9)
                ax.bar(x + width/2, user_scores, width, label='My Tone', color='#a394b8', alpha=0.9)
                
                ax.set_ylabel('Scores (0 - 100)', color='#e2e8f0', fontsize=9)
                ax.set_title('Tone Signature Comparison', color='#e2e8f0', fontsize=11, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(features, color='#e2e8f0', fontsize=9)
                ax.tick_params(colors='#e2e8f0', labelsize=8)
                ax.spines['bottom'].set_color('#2a2b36')
                ax.spines['left'].set_color('#2a2b36')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend(facecolor='#0b0c10', edgecolor='#2a2b36', labelcolor='#e2e8f0', fontsize=8)
                ax.grid(axis='y', linestyle='--', alpha=0.1, color='#ffffff')
                plt.ylim(0, 110)
                
                # Render chart inside a centered columns layout to respect figsize dimensions
                gcol1, gcol2, gcol3 = st.columns([1, 4, 1])
                with gcol2:
                    st.pyplot(fig, use_container_width=False)
                st.markdown("</div>", unsafe_allow_html=True)
                

                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
            finally:
                # Clean up temporary trimmed files
                if trimmed_ref_context_path and os.path.exists(trimmed_ref_context_path):
                    try:
                        os.unlink(trimmed_ref_context_path)
                    except Exception:
                        pass
                if trimmed_user_path and os.path.exists(trimmed_user_path):
                    try:
                        os.unlink(trimmed_user_path)
                    except Exception:
                        pass

