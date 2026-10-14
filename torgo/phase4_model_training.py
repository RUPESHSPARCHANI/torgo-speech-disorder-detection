"""
==============================================================================
Phase 4: Model Training
==============================================================================
Trains two classifiers on the preprocessed TORGO features:
  1. SVM with RBF kernel (class_weight='balanced')
  2. Random Forest with 100 estimators (class_weight='balanced')
Then runs 5-fold stratified cross-validation on both models.
"""

import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import RANDOM_STATE, N_FOLDS, N_ESTIMATORS
from utils import print_phase_header


def train_models(X_train, y_train, X_scaled, y, checklist):
    """
    Train SVM and Random Forest models with cross-validation.

    Args:
        X_train: Scaled training features
        y_train: Training labels
        X_scaled: Full scaled feature matrix (for cross-validation)
        y: Full label array (for cross-validation)
        checklist: PipelineChecklist instance

    Returns:
        dict with 'svm_model', 'rf_model', 'svm_cv_scores',
        'rf_cv_scores', 'svm_train_time', 'rf_train_time'
    """
    print_phase_header(4, "MODEL TRAINING")

    # ── Step 4.1: SVM (RBF kernel) ──────────────────────────────────────
    print("\n  Step 4.1: Training SVM (RBF kernel)...")
    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    )
    t0 = time.time()
    svm_model.fit(X_train, y_train)
    svm_train_time = time.time() - t0
    print(f"   SVM trained in {svm_train_time:.2f}s")
    checklist.mark_done("4.1", "Train SVM with RBF kernel")

    # ── Step 4.2: Random Forest ──────────────────────────────────────────
    print(f"\n  Step 4.2: Training Random Forest ({N_ESTIMATORS} estimators)...")
    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    print(f"   Random Forest trained in {rf_train_time:.2f}s")
    checklist.mark_done("4.2", f"Train Random Forest Classifier ({N_ESTIMATORS} estimators)")

    # ── Step 4.3: Cross-validation ───────────────────────────────────────
    print(f"\n  Step 4.3: Running {N_FOLDS}-fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    svm_cv_scores = cross_val_score(
        svm_model, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"   SVM  CV Accuracy: {svm_cv_scores.mean():.4f} +/- {svm_cv_scores.std():.4f}")
    print(f"   SVM  Fold scores: {[f'{s:.4f}' for s in svm_cv_scores]}")

    rf_cv_scores = cross_val_score(
        rf_model, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"   RF   CV Accuracy: {rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}")
    print(f"   RF   Fold scores: {[f'{s:.4f}' for s in rf_cv_scores]}")
    checklist.mark_done("4.3", f"{N_FOLDS}-fold Stratified Cross-Validation on both models")

    return {
        "svm_model": svm_model,
        "rf_model": rf_model,
        "svm_cv_scores": svm_cv_scores,
        "rf_cv_scores": rf_cv_scores,
        "svm_train_time": svm_train_time,
        "rf_train_time": rf_train_time,
    }


# ── Standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils import PipelineChecklist
    from phase1_data_loading import load_dataset
    from phase2_feature_extraction import run_feature_extraction
    from phase3_preprocessing import preprocess_data

    checklist = PipelineChecklist()
    df_files = load_dataset(checklist)
    df_features, y, df_meta = run_feature_extraction(df_files, checklist)
    X_train, X_test, y_train, y_test, X_scaled, y, feat_names = preprocess_data(
        df_features, y, df_meta, checklist
    )
    results = train_models(X_train, y_train, X_scaled, y, checklist)
    checklist.print_summary()
