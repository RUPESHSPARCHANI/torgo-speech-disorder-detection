"""
==============================================================================
market_loso_evaluation.py — Speaker-Independent (LOSO) Evaluation
==============================================================================
Evaluates models using Leave-One-Speaker-Out (LOSO) and Group-K-Fold cross-validation.
In real-world deployment, the model encounters speakers it has never heard before.
This module verifies true generalization across all 15 TORGO speakers.
"""

import sys
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

from config import FEATURES_CSV, OUTPUT_DIR, RANDOM_STATE
from utils import print_phase_header

sys.stdout.reconfigure(encoding="utf-8")


def run_speaker_independent_evaluation():
    print_phase_header("MARKET READY", "SPEAKER-INDEPENDENT (LOSO) EVALUATION")
    
    print("\n📂 Loading extracted features dataset...")
    df = pd.read_csv(FEATURES_CSV)
    
    # Extract clean speaker ID (strip trailing session e.g. FC01S01 -> FC01)
    df["speaker_id"] = df["speaker_session"].apply(lambda s: re.sub(r'S\d+$', '', str(s)))
    
    speakers = sorted(df["speaker_id"].unique())
    print(f"👥 Total Unique Speakers: {len(speakers)}")
    speaker_summary = df.groupby(["speaker_id", "category", "label_name"]).size().reset_index(name="sample_count")
    print(speaker_summary.to_string(index=False))
    
    # Separate features and target
    meta_cols = ["category", "gender", "speaker_session", "speaker_id", "label_name", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["speaker_id"].values
    
    print(f"\n📊 Feature Matrix: {X.shape[0]} samples, {X.shape[1]} acoustic features.")
    
    # ── 1. Leave-One-Speaker-Out (LOSO) Evaluation ─────────────────────────
    print("\n🔄 Running Leave-One-Speaker-Out (LOSO) Cross-Validation across all 15 speakers...")
    print("   (Training on 14 speakers, evaluating on 1 held-out unseen speaker)\n")
    
    logo = LeaveOneGroupOut()
    loso_results = []
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), 1):
        test_speaker = groups[test_idx[0]]
        test_category = df.iloc[test_idx[0]]["category"]
        test_label_name = df.iloc[test_idx[0]]["label_name"]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit scaler ONLY on training speakers to avoid data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM (RBF)
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
        svm.fit(X_train_scaled, y_train)
        svm_preds = svm.predict(X_test_scaled)
        svm_acc = accuracy_score(y_test, svm_preds)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        rf_preds = rf.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_preds)
        
        loso_results.append({
            "speaker_id": test_speaker,
            "category": test_category,
            "ground_truth": test_label_name,
            "samples": len(test_idx),
            "svm_accuracy": float(svm_acc),
            "rf_accuracy": float(rf_acc),
        })
        
        print(f"   [{fold:02d}/15] Held-out Speaker: {test_speaker:<6} ({test_label_name:<10}, {len(test_idx):>4} samples) "
              f"| SVM Acc: {svm_acc*100:>5.1f}% | RF Acc: {rf_acc*100:>5.1f}%")
        
    df_loso = pd.DataFrame(loso_results)
    
    # Overall Speaker-Independent Metrics (weighted by sample count)
    total_samples = df_loso["samples"].sum()
    svm_weighted_acc = (df_loso["svm_accuracy"] * df_loso["samples"]).sum() / total_samples
    rf_weighted_acc = (df_loso["rf_accuracy"] * df_loso["samples"]).sum() / total_samples
    
    print("\n" + "=" * 70)
    print("📈 SPEAKER-INDEPENDENT (LOSO) SUMMARY")
    print("=" * 70)
    print(f"  Overall SVM Speaker-Independent Accuracy:  {svm_weighted_acc * 100:.2f}%")
    print(f"  Overall RF  Speaker-Independent Accuracy:  {rf_weighted_acc * 100:.2f}%")
    print("=" * 70)
    
    # Save LOSO report to JSON
    report_file = OUTPUT_DIR / "speaker_independent_loso_report.json"
    with open(report_file, "w") as f:
        json.dump({
            "total_speakers": len(speakers),
            "total_samples": int(total_samples),
            "svm_weighted_accuracy": float(svm_weighted_acc),
            "rf_weighted_accuracy": float(rf_weighted_acc),
            "per_speaker_results": loso_results
        }, f, indent=2)
    print(f"💾 Report saved to: {report_file}")
    
    # ── 2. Visualization: Speaker-by-Speaker Generalization Plot ─────────
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "axes.edgecolor": "#333366",
        "font.family": "sans-serif",
    })
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Speaker-Independent (LOSO) Generalization Across Unseen Individuals", fontsize=15, fontweight="bold", color="#00d2ff")
    
    x = np.arange(len(df_loso))
    width = 0.38
    
    rects1 = ax.bar(x - width/2, df_loso["svm_accuracy"] * 100, width, label="SVM (RBF)", color="#ff6b6b", edgecolor="#333366")
    rects2 = ax.bar(x + width/2, df_loso["rf_accuracy"] * 100, width, label="Random Forest", color="#00d2ff", edgecolor="#333366")
    
    # Baseline line
    ax.axhline(svm_weighted_acc * 100, color="#ff6b6b", linestyle="--", alpha=0.7, label=f"SVM Mean ({svm_weighted_acc*100:.1f}%)")
    ax.axhline(rf_weighted_acc * 100, color="#00d2ff", linestyle="--", alpha=0.7, label=f"RF Mean ({rf_weighted_acc*100:.1f}%)")
    
    ax.set_ylabel("Accuracy on Unseen Speaker (%)", fontsize=12)
    ax.set_xlabel("Held-Out Speaker ID (Ground Truth Class)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row.speaker_id}\n({row.ground_truth[:4]})" for row in df_loso.itertuples()], fontsize=10)
    ax.set_ylim([0, 115])
    ax.legend(loc="upper right", facecolor="#16213e", edgecolor="#333366")
    
    plt.tight_layout()
    plot_file = OUTPUT_DIR / "06_loso_speaker_evaluation.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Visualization saved to: {plot_file}")
    
    return df_loso


if __name__ == "__main__":
    run_speaker_independent_evaluation()
