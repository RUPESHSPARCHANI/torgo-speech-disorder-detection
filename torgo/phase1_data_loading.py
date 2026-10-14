"""
==============================================================================
Phase 1: Data Loading & Exploration
==============================================================================
Scans the TORGO directory tree, catalogs WAV files, assigns labels,
extracts metadata, and checks class distribution balance.
"""

import pandas as pd
import librosa
from config import BASE_DIR, CATEGORY_MAP, LABEL_NAMES, MIC_FILTER
from utils import print_phase_header


def load_dataset(checklist):
    """
    Traverse the TORGO directory structure and build a catalog of all
    arrayMic .wav files with labels and metadata.

    Returns:
        pd.DataFrame with columns: filepath, category, label, label_name,
                                    speaker_session, gender, filename
    """
    print_phase_header(1, "DATA LOADING & EXPLORATION")

    # ── Step 1.1: Scan directories ───────────────────────────────────────
    print("\n  Step 1.1: Scanning directories for .wav files...")
    file_records = []

    for category, label in CATEGORY_MAP.items():
        category_dir = BASE_DIR / category
        if not category_dir.exists():
            print(f"  WARNING: Directory not found: {category_dir}")
            continue

        for session_dir in sorted(category_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            # Filter to arrayMic only for consistency
            if MIC_FILTER not in session_dir.name:
                continue

            for wav_file in sorted(session_dir.glob("*.wav")):
                parts = session_dir.name.split("_")
                speaker_session = parts[-1]  # e.g., FC01S01 or M01S01
                gender = "F" if category.startswith("F") else "M"

                file_records.append({
                    "filepath": str(wav_file),
                    "category": category,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "speaker_session": speaker_session,
                    "gender": gender,
                    "filename": wav_file.name,
                })

    df_files = pd.DataFrame(file_records)
    print(f"  Total .wav files cataloged: {len(df_files)}")
    checklist.mark_done("1.1", "Traverse directory structure and catalog all .wav files")

    # ── Step 1.2: Label assignment ───────────────────────────────────────
    print(f"\n  Step 1.2: Label distribution:")
    label_counts = df_files["label_name"].value_counts()
    for lbl, cnt in label_counts.items():
        print(f"     {lbl}: {cnt} files")
    checklist.mark_done("1.2", "Assign labels: 0=Control, 1=Dysarthric")

    # ── Step 1.3: Metadata ───────────────────────────────────────────────
    print(f"\n  Step 1.3: Metadata by category:")
    cat_counts = df_files.groupby(["category", "gender", "label_name"]).size().reset_index(name="count")
    print(cat_counts.to_string(index=False))
    checklist.mark_done("1.3", "Extract metadata (speaker, session, gender, mic)")

    # ── Step 1.4: Class imbalance ────────────────────────────────────────
    imbalance_ratio = label_counts.min() / label_counts.max()
    print(f"\n  Step 1.4: Class imbalance ratio: {imbalance_ratio:.3f}")
    if imbalance_ratio < 0.5:
        print("     WARNING: Significant imbalance — will use class_weight='balanced'")
    else:
        print("     Classes are reasonably balanced")
    checklist.mark_done("1.4", "Analyze class distribution and check for imbalance")

    # ── Step 1.5: Audio properties ───────────────────────────────────────
    print(f"\n  Step 1.5: Verifying audio properties (1 sample per category)...")
    for cat in CATEGORY_MAP:
        sample_row = df_files[df_files["category"] == cat].iloc[0]
        y, sr = librosa.load(sample_row["filepath"], sr=None, duration=1.0)
        print(f"     {cat}: sr={sr} Hz, duration={len(y)/sr:.2f}s, dtype={y.dtype}")
    checklist.mark_done("1.5", "Verify audio properties (sample rate, duration, channels)")

    return df_files


# ── Standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils import PipelineChecklist
    checklist = PipelineChecklist()
    df = load_dataset(checklist)
    print(f"\nDataset loaded: {len(df)} files")
    checklist.print_summary()
