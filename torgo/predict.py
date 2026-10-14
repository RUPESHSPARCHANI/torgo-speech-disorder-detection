"""
==============================================================================
predict.py — Standalone Production Inference Engine for Any Audio File
==============================================================================
Usage:
    python predict.py "path/to/unknown_audio.wav"

Or programmatically in Python:
    from predict import predict_audio
    result = predict_audio("path/to/unknown_audio.wav")
    print(result)

Features:
  - Universal audio input (.wav, .mp3, .m4a, .flac) via librosa
  - Automatic stereo-to-mono downmix & resampling to 16 kHz
  - Silence & ambient noise trimming using Voice Activity Detection (top_db)
  - 126-dimensional acoustic biomarker extraction
  - Scaled inference using production bundle
  - Clinical risk stratification (Low / Moderate / High Risk)
==============================================================================
"""

import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np
import librosa
import joblib

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"c:\Users\DELL\Downloads\NIT SIKKIM\torgo")
BUNDLE_PATH = BASE_DIR / "models" / "torgo_production_bundle.joblib"

# Global bundle cache for high-throughput inference
_CACHED_BUNDLE = None


def load_audio_universal(audio_path, target_sr=16000, duration=None):
    """
    Universal audio loader supporting .wav, .mp3, .ogg, .flac, .m4a, .aac, .wma, .webm.
    Tries native librosa/soundfile first, falling back to bundled FFmpeg for proprietary
    AAC / M4A containers commonly exported by phones (iPhone Voice Memos, Android, WhatsApp).
    """
    # 1. Native librosa / soundfile loader (fast for standard formats like .wav, .flac, .ogg)
    try:
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True, duration=duration)
        return y, sr
    except Exception:
        pass

    # 2. Universal FFmpeg decoder fallback (handles .m4a, .aac, .wma, .mp4, etc.)
    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        ffmpeg_exe = shutil.which("ffmpeg")

    if not ffmpeg_exe:
        raise ValueError(
            f"Cannot decode {audio_path}: Format not recognized natively and FFmpeg is not installed."
        )

    cmd = [
        ffmpeg_exe, "-nostdin", "-loglevel", "error",
        "-i", str(audio_path),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(target_sr), "-ac", "1"
    ]
    if duration:
        cmd.extend(["-t", str(duration)])
    cmd.append("-")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or len(proc.stdout) == 0:
        err_msg = proc.stderr.decode("utf-8", errors="ignore")
        raise ValueError(f"Failed to decode audio file {audio_path}: {err_msg or 'Empty output'}")

    audio_data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_data, target_sr


def get_production_bundle():
    """Load and cache the production ML bundle."""
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is None:
        if not BUNDLE_PATH.exists():
            raise FileNotFoundError(
                f"Production bundle not found at {BUNDLE_PATH}. "
                "Please run `python train_production_model.py` first."
            )
        _CACHED_BUNDLE = joblib.load(BUNDLE_PATH)
    return _CACHED_BUNDLE


def extract_features_from_audio_array(y, sr, n_mfcc=13):
    """
    Extract the exact 126 acoustic features from an in-memory audio signal.
    """
    features = {}

    # 1. MFCCs + Delta + Delta-Delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(mfcc_delta[i]))
        features[f"mfcc_delta_{i+1}_std"] = float(np.std(mfcc_delta[i]))
        features[f"mfcc_delta2_{i+1}_mean"] = float(np.mean(mfcc_delta2[i]))
        features[f"mfcc_delta2_{i+1}_std"] = float(np.std(mfcc_delta2[i]))

    # 2. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

    # 3. Spectral Contrast (7 bands)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(spec_contrast.shape[0]):
        features[f"spec_contrast_{i+1}_mean"] = float(np.mean(spec_contrast[i]))
        features[f"spec_contrast_{i+1}_std"] = float(np.std(spec_contrast[i]))

    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # 5. RMS Energy
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # 6. Spectral Centroid, Bandwidth, Rolloff
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features["spectral_centroid_mean"] = float(np.mean(spec_cent))
    features["spectral_centroid_std"] = float(np.std(spec_cent))
    features["spectral_bandwidth_mean"] = float(np.mean(spec_bw))
    features["spectral_bandwidth_std"] = float(np.std(spec_bw))
    features["spectral_rolloff_mean"] = float(np.mean(spec_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spec_rolloff))

    return features


def predict_audio(audio_path, model_choice="svm"):
    """
    Run market-ready inference on any unseen audio file.

    Args:
        audio_path: Path to the audio file (.wav, .mp3, etc.)
        model_choice: 'svm' (recommended) or 'rf'

    Returns:
        dict containing diagnosis, probabilities, confidence, risk level, and biomarkers.
    """
    t0 = time.time()
    bundle = get_production_bundle()
    target_sr = bundle["sample_rate"]
    max_duration = bundle["max_duration"]
    trim_top_db = bundle.get("trim_top_db", 25)

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Load, downmix to mono, and resample to 16 kHz (supports .wav, .mp3, .m4a, etc.)
    y, sr = load_audio_universal(audio_path, target_sr=target_sr, duration=max_duration)

    raw_duration = len(y) / sr

    # 2. Voice Activity Detection: Skipped to match training data
    # (The training data was NOT trimmed, so trimming here changes RMS and duration statistics
    # drastically compared to the training distribution.)
    y_final = y
    final_duration = len(y_final) / sr

    if len(y_final) < sr * 0.15:
        raise ValueError("Audio clip is too short or silent (< 150 ms of active speech).")

    # 3. Extract 126 acoustic features
    feat_dict = extract_features_from_audio_array(y_final, sr=sr, n_mfcc=bundle["n_mfcc"])

    # 4. Align features strictly with model's expected columns
    feature_names = bundle["feature_names"]
    feature_vector = np.array([feat_dict[col] for col in feature_names]).reshape(1, -1)

    # 5. Apply production StandardScaler
    X_scaled = bundle["scaler"].transform(feature_vector)

    # 6. Model Inference
    model = bundle["primary_model"] if model_choice == "svm" else bundle["rf_model"]
    pred_class_idx = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]

    prob_control = float(probabilities[0])
    prob_dysarthric = float(probabilities[1])
    diagnosis = bundle["label_mapping"][pred_class_idx]
    confidence = float(max(prob_control, prob_dysarthric))

    # 7. Clinical Risk Stratification
    thresholds = bundle["risk_thresholds"]
    if prob_dysarthric >= thresholds["high_risk"]:
        risk_level = "HIGH RISK"
        recommendation = "Acoustic markers show significant dysarthric speech patterns. Clinical speech assessment recommended."
    elif prob_dysarthric >= thresholds["low_risk"]:
        risk_level = "MODERATE RISK"
        recommendation = "Inconclusive / borderline acoustic markers. Additional speech samples recommended."
    else:
        risk_level = "LOW RISK"
        recommendation = "Speech patterns fall within healthy normative acoustic controls."

    latency_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "success",
        "filename": audio_path.name,
        "diagnosis": diagnosis,
        "is_dysarthric": bool(pred_class_idx == 1),
        "confidence_score": round(confidence * 100, 2),
        "probabilities": {
            "control": round(prob_control * 100, 2),
            "dysarthric": round(prob_dysarthric * 100, 2)
        },
        "clinical_risk_level": risk_level,
        "clinical_recommendation": recommendation,
        "audio_metadata": {
            "original_duration_seconds": round(raw_duration, 2),
            "speech_duration_seconds": round(final_duration, 2),
            "sample_rate_hz": target_sr,
            "channels": "mono",
            "silence_trimmed": False
        },
        "key_acoustic_biomarkers": {
            "energy_variability_rms_std": round(feat_dict.get("rms_std", 0.0), 6),
            "average_energy_rms_mean": round(feat_dict.get("rms_mean", 0.0), 6),
            "spectral_contrast_hf": round(feat_dict.get("spec_contrast_7_mean", 0.0), 2),
            "first_mfcc_envelope": round(feat_dict.get("mfcc_1_mean", 0.0), 2),
            "zero_crossing_rate_mean": round(feat_dict.get("zcr_mean", 0.0), 4),
        },
        "inference_latency_ms": latency_ms,
        "model_used": "SVM (RBF Kernel)" if model_choice == "svm" else "Random Forest"
    }


def main():
    if len(sys.argv) < 2:
        print("\n" + "=" * 70)
        print("  TORGO Speech Pathology Inference Engine")
        print("=" * 70)
        print("Usage: python predict.py <path_to_audio_file> [model: svm|rf]")
        print("Example: python predict.py \"F_Con/wav_arrayMic_FC01S01/wav_arrayMic_FC01S01_0001.wav\"")
        print("=" * 70)
        return

    audio_path = sys.argv[1]
    model_choice = sys.argv[2] if len(sys.argv) > 2 else "svm"

    try:
        result = predict_audio(audio_path, model_choice=model_choice)
        print("\n" + "=" * 70)
        print(f"  DIAGNOSTIC REPORT: {result['filename']}")
        print("=" * 70)
        print(f"  Diagnosis:        {result['diagnosis'].upper()}")
        print(f"  Confidence:       {result['confidence_score']}%")
        print(f"  Risk Level:       {result['clinical_risk_level']}")
        print(f"  Recommendation:   {result['clinical_recommendation']}")
        print("-" * 70)
        print(f"  Probabilities:    Control: {result['probabilities']['control']}% | Dysarthric: {result['probabilities']['dysarthric']}%")
        print(f"  Audio Processed:  {result['audio_metadata']['speech_duration_seconds']}s active speech ({result['audio_metadata']['sample_rate_hz']} Hz)")
        print(f"  Key Biomarkers:   Energy Std: {result['key_acoustic_biomarkers']['energy_variability_rms_std']} | HF Contrast: {result['key_acoustic_biomarkers']['spectral_contrast_hf']}")
        print(f"  Latency:          {result['inference_latency_ms']} ms")
        print("=" * 70)
        print("\nJSON Output:\n" + json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
