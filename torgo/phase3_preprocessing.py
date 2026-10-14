"""
==============================================================================
Phase 3: Data Preprocessing
==============================================================================
Cleans the feature matrix (NaN/Inf handling), applies StandardScaler,
checks class imbalance, and performs stratified train/test split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import TEST_SIZE, RANDOM_STATE, LABEL_NAMES
from utils import print_phase_header


def preprocess_data(df_features, y, df_meta, checklist):
    """
    Clean, scale, and split the feature matrix.

    Args:
        df_features: DataFrame of extracted audio features
        y: numpy array of labels
        df_meta: DataFrame of metadata
        checklist: PipelineChecklist instance

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print_phase_header(3, "DATA PREPROCESSING")

    # ── Step 3.1: Handle NaN values ──────────────────────────────────────
    nan_count = df_features.isna().sum().sum()
    print(f"\n  Step 3.1: NaN values found: {nan_count}")
    if nan_count > 0:
        valid_mask = ~df_features.isna().any(axis=1)
        df_features = df_features[valid_mask].reset_index(drop=True)
        y = y[valid_mask.values]
        df_meta = df_meta[valid_mask].reset_index(drop=True)
        print(f"   Dropped {(~valid_mask).sum()} rows with NaN values")
        print(f"   Remaining samples: {len(df_features)}")
    checklist.mark_done("3.1", "Handle missing/NaN values")

    # Also replace infinite values
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    inf_count = df_features.isna().sum().sum()
    if inf_count > 0:
        valid_mask = ~df_features.isna().any(axis=1)
        df_features = df_features[valid_mask].reset_index(drop=True)
        y = y[valid_mask.values]
        df_meta = df_meta[valid_mask].reset_index(drop=True)
        print(f"   Dropped {(~valid_mask).sum()} rows with Inf values")

    feature_names = df_features.columns.tolist()
    X = df_features.values

    # ── Step 3.2: Feature scaling ────────────────────────────────────────
    print(f"\n  Step 3.2: Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Mean after scaling: {X_scaled.mean():.6f} (target: ~0)")
    print(f"   Std after scaling:  {X_scaled.std():.6f} (target: ~1)")
    checklist.mark_done("3.2", "Feature scaling using StandardScaler")

    # ── Step 3.3: Class imbalance ────────────────────────────────────────
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Step 3.3: Post-cleaning class distribution:")
    for u, c in zip(unique, counts):
        print(f"   {LABEL_NAMES[u]}: {c} samples ({c/len(y)*100:.1f}%)")
    print(f"   Strategy: Using class_weight='balanced' for both models")
    checklist.mark_done("3.3", "Check for class imbalance and plan mitigation")

    # ── Step 3.4: Stratified train/test split ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n  Step 3.4: Stratified train/test split "
          f"({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set:     {X_test.shape[0]} samples")
    print(f"   Train labels: Control={sum(y_train==0)}, Dysarthric={sum(y_train==1)}")
    print(f"   Test labels:  Control={sum(y_test==0)}, Dysarthric={sum(y_test==1)}")
    checklist.mark_done("3.4", "Stratified Train/Test split (80/20)")

    return X_train, X_test, y_train, y_test, X_scaled, y, feature_names


# ── Standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils import PipelineChecklist
    from phase1_data_loading import load_dataset
    from phase2_feature_extraction import run_feature_extraction

    checklist = PipelineChecklist()
    df_files = load_dataset(checklist)
    df_features, y, df_meta = run_feature_extraction(df_files, checklist)
    results = preprocess_data(df_features, y, df_meta, checklist)
    checklist.print_summary()
