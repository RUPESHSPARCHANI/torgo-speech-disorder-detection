"""
==============================================================================
Phase 2: Audio Feature Extraction
==============================================================================
Extracts a comprehensive set of audio features from each WAV file:
  - MFCCs (13) + Delta + Delta-Delta = 39 x 2 (mean+std) = 78 features
  - Chroma (12) x 2 = 24 features
  - Spectral Contrast (7) x 2 = 14 features
  - ZCR x 2 = 2 features
  - RMS Energy x 2 = 2 features
  - Spectral Centroid, Bandwidth, Rolloff x 2 = 6 features
  Total: ~126 features per audio sample
"""

import time
import numpy as np
import pandas as pd
import librosa
from config import SAMPLE_RATE, N_MFCC, MAX_DURATION, OUTPUT_DIR, FEATURES_CSV
from utils import print_phase_header


def extract_features_from_file(filepath, sr=SAMPLE_RATE, max_duration=MAX_DURATION):
    """
    Extract a comprehensive set of audio features from a single .wav file.

    Args:
        filepath: Path to the .wav file
        sr: Target sample rate
        max_duration: Max seconds to load

    Returns:
        dict of feature_name -> value, or None on failure
    """
    try:
        y, sr = librosa.load(filepath, sr=sr, duration=max_duration)

        # Skip very short audio (< 100ms)
        if len(y) < sr * 0.1:
            return None

        features = {}

        # ── MFCCs + Delta + Delta-Delta ──────────────────────────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        for i in range(N_MFCC):
            features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
            features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])
            features[f"mfcc_delta_{i+1}_mean"] = np.mean(mfcc_delta[i])
            features[f"mfcc_delta_{i+1}_std"] = np.std(mfcc_delta[i])
            features[f"mfcc_delta2_{i+1}_mean"] = np.mean(mfcc_delta2[i])
            features[f"mfcc_delta2_{i+1}_std"] = np.std(mfcc_delta2[i])

        # ── Chroma ───────────────────────────────────────────────────────
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f"chroma_{i+1}_mean"] = np.mean(chroma[i])
            features[f"chroma_{i+1}_std"] = np.std(chroma[i])

        # ── Spectral Contrast ────────────────────────────────────────────
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spec_contrast.shape[0]):
            features[f"spec_contrast_{i+1}_mean"] = np.mean(spec_contrast[i])
            features[f"spec_contrast_{i+1}_std"] = np.std(spec_contrast[i])

        # ── Zero Crossing Rate ───────────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = np.mean(zcr)
        features["zcr_std"] = np.std(zcr)

        # ── RMS Energy ───────────────────────────────────────────────────
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = np.mean(rms)
        features["rms_std"] = np.std(rms)

        # ── Spectral Centroid, Bandwidth, Rolloff ────────────────────────
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        features["spectral_centroid_mean"] = np.mean(spec_cent)
        features["spectral_centroid_std"] = np.std(spec_cent)
        features["spectral_bandwidth_mean"] = np.mean(spec_bw)
        features["spectral_bandwidth_std"] = np.std(spec_bw)
        features["spectral_rolloff_mean"] = np.mean(spec_rolloff)
        features["spectral_rolloff_std"] = np.std(spec_rolloff)

        return features

    except Exception as e:
        print(f"  ERROR processing {filepath}: {e}")
        return None


def run_feature_extraction(df_files, checklist):
    """
    Extract features from all audio files in the dataset catalog.

    Args:
        df_files: DataFrame from Phase 1 with 'filepath' and 'label' columns
        checklist: PipelineChecklist instance

    Returns:
        df_features (pd.DataFrame), labels (np.array), df_meta (pd.DataFrame)
    """
    print_phase_header(2, "AUDIO FEATURE EXTRACTION")

    print(f"\n  Extracting features from {len(df_files)} audio files...")
    print(f"  This may take several minutes...\n")

    all_features = []
    labels = []
    metadata_list = []
    errors = 0

    start_time = time.time()
    total = len(df_files)

    for idx, row in df_files.iterrows():
        feats = extract_features_from_file(row["filepath"])
        if feats is not None:
            all_features.append(feats)
            labels.append(row["label"])
            metadata_list.append({
                "category": row["category"],
                "gender": row["gender"],
                "speaker_session": row["speaker_session"],
                "label_name": row["label_name"],
            })
        else:
            errors += 1

        # Progress reporting every 500 files
        processed = idx + 1
        if processed % 500 == 0 or processed == total:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"   Progress: {processed}/{total} ({processed/total*100:.1f}%) "
                  f"| Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s | Errors: {errors}")

    total_time = time.time() - start_time
    print(f"\n  Feature extraction complete in {total_time:.1f}s")
    print(f"  Successful: {len(all_features)} / {total}")
    print(f"  Errors/skipped: {errors}")

    checklist.mark_done("2.1", "Extract MFCCs (13 coefficients + delta + delta-delta)")
    checklist.mark_done("2.2", "Extract Chroma features (12 pitch classes)")
    checklist.mark_done("2.3", "Extract Spectral Contrast (7 bands)")
    checklist.mark_done("2.4", "Extract Zero Crossing Rate")
    checklist.mark_done("2.5", "Extract RMS Energy")
    checklist.mark_done("2.6", "Extract Spectral Centroid, Bandwidth, Rolloff")
    checklist.mark_done("2.7", "Aggregate features per file (mean + std)")

    # Build DataFrames
    df_features = pd.DataFrame(all_features)
    df_meta = pd.DataFrame(metadata_list)
    y = np.array(labels)

    print(f"\n  Feature matrix shape: {df_features.shape}")
    print(f"  Features per sample:  {df_features.shape[1]}")

    # Save to CSV for reusability
    df_save = pd.concat([df_meta, df_features], axis=1)
    df_save["label"] = y
    df_save.to_csv(FEATURES_CSV, index=False)
    print(f"  Saved to: {FEATURES_CSV}")

    return df_features, y, df_meta


# ── Standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils import PipelineChecklist
    from phase1_data_loading import load_dataset

    checklist = PipelineChecklist()
    df_files = load_dataset(checklist)
    df_features, y, df_meta = run_feature_extraction(df_files, checklist)
    checklist.print_summary()
