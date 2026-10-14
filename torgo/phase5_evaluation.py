"""
==============================================================================
Phase 5: Model Evaluation & Comparison
==============================================================================
Evaluates both SVM and Random Forest on the test set:
  - Classification Reports (Precision, Recall, F1)
  - Confusion Matrices
  - ROC-AUC Scores
  - Head-to-head comparison table
  - Random Forest feature importances
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)

from config import TARGET_NAMES
from utils import print_phase_header


def evaluate_models(models_dict, X_test, y_test, feature_names, checklist):
    """
    Run full evaluation on both trained models.

    Args:
        models_dict: dict from Phase 4 with models and CV scores
        X_test: Scaled test features
        y_test: Test labels
        feature_names: list of feature column names
        checklist: PipelineChecklist instance

    Returns:
        dict with all evaluation metrics and predictions
    """
    print_phase_header(5, "MODEL EVALUATION & COMPARISON")

    svm_model = models_dict["svm_model"]
    rf_model = models_dict["rf_model"]

    # Predictions
    svm_y_pred = svm_model.predict(X_test)
    svm_y_proba = svm_model.predict_proba(X_test)[:, 1]
    rf_y_pred = rf_model.predict(X_test)
    rf_y_proba = rf_model.predict_proba(X_test)[:, 1]

    # ── Step 5.1: Classification Reports ─────────────────────────────────
    print("\n  Step 5.1: Classification Reports")
    print("\n" + "-" * 50)
    print("  SVM Classification Report:")
    print("-" * 50)
    svm_report = classification_report(y_test, svm_y_pred, target_names=TARGET_NAMES)
    print(svm_report)

    print("-" * 50)
    print("  Random Forest Classification Report:")
    print("-" * 50)
    rf_report = classification_report(y_test, rf_y_pred, target_names=TARGET_NAMES)
    print(rf_report)
    checklist.mark_done("5.1", "Generate Classification Reports (Precision, Recall, F1)")

    # ── Step 5.2: Confusion Matrices ─────────────────────────────────────
    svm_cm = confusion_matrix(y_test, svm_y_pred)
    rf_cm = confusion_matrix(y_test, rf_y_pred)
    print(f"  Step 5.2: Confusion Matrices")
    print(f"   SVM:  {svm_cm.tolist()}")
    print(f"   RF:   {rf_cm.tolist()}")
    checklist.mark_done("5.2", "Generate Confusion Matrices")

    # ── Step 5.3: ROC-AUC Scores ─────────────────────────────────────────
    svm_auc = roc_auc_score(y_test, svm_y_proba)
    rf_auc = roc_auc_score(y_test, rf_y_proba)
    print(f"\n  Step 5.3: ROC-AUC Scores")
    print(f"   SVM ROC-AUC: {svm_auc:.4f}")
    print(f"   RF  ROC-AUC: {rf_auc:.4f}")
    checklist.mark_done("5.3", "Compute ROC-AUC scores")

    # ── Step 5.4: Comparison ─────────────────────────────────────────────
    svm_acc = accuracy_score(y_test, svm_y_pred)
    rf_acc = accuracy_score(y_test, rf_y_pred)
    svm_f1 = f1_score(y_test, svm_y_pred, average="weighted")
    rf_f1 = f1_score(y_test, rf_y_pred, average="weighted")

    svm_cv_mean = models_dict["svm_cv_scores"].mean()
    rf_cv_mean = models_dict["rf_cv_scores"].mean()

    print(f"\n  Step 5.4: Model Comparison Summary")
    print(f"   {'Metric':<25} {'SVM':>10} {'Random Forest':>15}")
    print(f"   {'-'*25} {'-'*10} {'-'*15}")
    print(f"   {'Accuracy':<25} {svm_acc:>10.4f} {rf_acc:>15.4f}")
    print(f"   {'F1 (weighted)':<25} {svm_f1:>10.4f} {rf_f1:>15.4f}")
    print(f"   {'ROC-AUC':<25} {svm_auc:>10.4f} {rf_auc:>15.4f}")
    print(f"   {'CV Accuracy (mean)':<25} {svm_cv_mean:>10.4f} {rf_cv_mean:>15.4f}")
    print(f"   {'Training Time (s)':<25} {models_dict['svm_train_time']:>10.2f} "
          f"{models_dict['rf_train_time']:>15.2f}")

    best_model_name = "SVM" if svm_f1 > rf_f1 else "Random Forest"
    print(f"\n   Best performing model: {best_model_name}")
    checklist.mark_done("5.4", "Compare SVM vs Random Forest performance")

    # ── Step 5.5: Feature Importances ────────────────────────────────────
    print(f"\n  Step 5.5: Top 20 Feature Importances (Random Forest)")
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    for _, row in importance_df.head(20).iterrows():
        print(f"   {row['feature']:<35} {row['importance']:.6f}")
    checklist.mark_done("5.5", "Analyze feature importances (Random Forest)")

    # ROC curve data
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_y_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_proba)

    return {
        "svm_y_pred": svm_y_pred,
        "svm_y_proba": svm_y_proba,
        "rf_y_pred": rf_y_pred,
        "rf_y_proba": rf_y_proba,
        "svm_cm": svm_cm,
        "rf_cm": rf_cm,
        "svm_auc": svm_auc,
        "rf_auc": rf_auc,
        "svm_acc": svm_acc,
        "rf_acc": rf_acc,
        "svm_f1": svm_f1,
        "rf_f1": rf_f1,
        "svm_cv_mean": svm_cv_mean,
        "rf_cv_mean": rf_cv_mean,
        "svm_fpr": svm_fpr,
        "svm_tpr": svm_tpr,
        "rf_fpr": rf_fpr,
        "rf_tpr": rf_tpr,
        "importance_df": importance_df,
        "best_model_name": best_model_name,
    }


# ── Standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils import PipelineChecklist
    from phase1_data_loading import load_dataset
    from phase2_feature_extraction import run_feature_extraction
    from phase3_preprocessing import preprocess_data
    from phase4_model_training import train_models

    checklist = PipelineChecklist()
    df_files = load_dataset(checklist)
    df_features, y, df_meta = run_feature_extraction(df_files, checklist)
    X_train, X_test, y_train, y_test, X_scaled, y, feat_names = preprocess_data(
        df_features, y, df_meta, checklist
    )
    models = train_models(X_train, y_train, X_scaled, y, checklist)
    eval_results = evaluate_models(models, X_test, y_test, feat_names, checklist)
    checklist.print_summary()
