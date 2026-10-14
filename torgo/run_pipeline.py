"""
==============================================================================
run_pipeline.py — TORGO ML Pipeline Orchestrator
==============================================================================
Main entry point that runs all 6 phases in sequence:
  Phase 1: Data Loading & Exploration
  Phase 2: Audio Feature Extraction
  Phase 3: Data Preprocessing
  Phase 4: Model Training (SVM + Random Forest)
  Phase 5: Model Evaluation & Comparison
  Phase 6: Visualization & Reporting

Usage:
    python run_pipeline.py
==============================================================================
"""

import time
from config import OUTPUT_DIR
from utils import PipelineChecklist

# Import all phases
from phase1_data_loading import load_dataset
from phase2_feature_extraction import run_feature_extraction
from phase3_preprocessing import preprocess_data
from phase4_model_training import train_models
from phase5_evaluation import evaluate_models
from phase6_visualization import generate_visualizations


def main():
    """Execute the complete TORGO ML pipeline."""
    print("=" * 70)
    print("  TORGO DYSARTHRIC SPEECH CLASSIFICATION PIPELINE")
    print("  Task: Binary Classification (Dysarthric vs Control)")
    print("  Models: SVM (RBF) + Random Forest")
    print("=" * 70)

    pipeline_start = time.time()
    checklist = PipelineChecklist()

    # ── Phase 1: Data Loading ────────────────────────────────────────────
    df_files = load_dataset(checklist)

    # ── Phase 2: Feature Extraction ──────────────────────────────────────
    df_features, y, df_meta = run_feature_extraction(df_files, checklist)

    # ── Phase 3: Preprocessing ───────────────────────────────────────────
    X_train, X_test, y_train, y_test, X_scaled, y_clean, feature_names = \
        preprocess_data(df_features, y, df_meta, checklist)

    # ── Phase 4: Model Training ──────────────────────────────────────────
    models_dict = train_models(X_train, y_train, X_scaled, y_clean, checklist)

    # ── Phase 5: Evaluation ──────────────────────────────────────────────
    eval_results = evaluate_models(
        models_dict, X_test, y_test, feature_names, checklist
    )

    # ── Phase 6: Visualization ───────────────────────────────────────────
    generate_visualizations(
        df_files, y_clean, eval_results, models_dict, checklist
    )

    # ── Final Summary ────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    checklist.print_summary()

    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"""
  Dataset:         TORGO Dysarthric Speech Database
  Task:            Binary Classification (Dysarthric vs Control)
  Total Samples:   {len(y_clean)}
  Features:        {len(feature_names)} audio features per sample
  Train/Test:      {X_train.shape[0]} / {X_test.shape[0]}

  +------------------+--------------+------------------+
  | Metric           | SVM (RBF)    | Random Forest    |
  +------------------+--------------+------------------+
  | Accuracy         | {eval_results['svm_acc']:>10.4f}   | {eval_results['rf_acc']:>14.4f}   |
  | F1 (weighted)    | {eval_results['svm_f1']:>10.4f}   | {eval_results['rf_f1']:>14.4f}   |
  | ROC-AUC          | {eval_results['svm_auc']:>10.4f}   | {eval_results['rf_auc']:>14.4f}   |
  | CV Acc (5-fold)  | {eval_results['svm_cv_mean']:>10.4f}   | {eval_results['rf_cv_mean']:>14.4f}   |
  | Training Time    | {models_dict['svm_train_time']:>8.2f}s   | {models_dict['rf_train_time']:>12.2f}s   |
  +------------------+--------------+------------------+

  Best Model: {eval_results['best_model_name']}
  Total Pipeline Time: {total_time:.1f}s

  Output files saved to: {OUTPUT_DIR}
     - extracted_features.csv
     - 01_class_distribution.png
     - 02_confusion_matrices.png
     - 03_roc_curves.png
     - 04_feature_importances.png
     - 05_model_comparison.png
""")

    print("=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
