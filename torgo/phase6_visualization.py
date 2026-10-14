"""
==============================================================================
Phase 6: Visualization & Reporting
==============================================================================
Generates publication-quality plots and saves them to the results/ folder:
  1. Class distribution (overall + by category)
  2. Confusion matrix heatmaps (SVM vs RF)
  3. ROC curves with AUC
  4. Top 20 feature importances
  5. Model comparison bar chart
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR
from utils import print_phase_header


# Global dark-theme plot style
PLOT_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333366",
    "font.family": "sans-serif",
    "font.size": 11,
}


def generate_visualizations(df_files, y_full, eval_results, models_dict, checklist):
    """
    Generate and save all visualization plots.

    Args:
        df_files: Original file catalog DataFrame
        y_full: Full label array (post-cleaning)
        eval_results: dict from Phase 5 with metrics and predictions
        models_dict: dict from Phase 4 with training times and CV scores
        checklist: PipelineChecklist instance
    """
    print_phase_header(6, "VISUALIZATION & REPORTING")
    plt.rcParams.update(PLOT_STYLE)

    # ── 6.1: Class Distribution ──────────────────────────────────────────
    print("\n  Step 6.1: Plotting class distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TORGO Dataset - Class Distribution",
                 fontsize=16, fontweight="bold", color="#00d2ff")

    colors = ["#00d2ff", "#ff6b6b"]

    # Overall
    class_labels = ["Control", "Dysarthric"]
    class_counts = [sum(y_full == 0), sum(y_full == 1)]
    bars = axes[0].bar(class_labels, class_counts, color=colors,
                       edgecolor="#333366", linewidth=1.5)
    axes[0].set_title("Overall Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Samples")
    for bar, cnt in zip(bars, class_counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     str(cnt), ha="center", va="bottom", fontsize=12,
                     fontweight="bold", color="#ffffff")

    # By category
    cat_data = df_files["category"].value_counts().sort_index()
    cat_colors = ["#00d2ff", "#ff6b6b", "#00d2ff", "#ff6b6b"]
    bars2 = axes[1].bar(cat_data.index, cat_data.values, color=cat_colors,
                        edgecolor="#333366", linewidth=1.5)
    axes[1].set_title("Distribution by Category", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Number of Files")
    for bar, cnt in zip(bars2, cat_data.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     str(cnt), ha="center", va="bottom", fontsize=11,
                     fontweight="bold", color="#ffffff")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {OUTPUT_DIR / '01_class_distribution.png'}")
    checklist.mark_done("6.1", "Plot class distribution")

    # ── 6.2: Confusion Matrices ──────────────────────────────────────────
    print("\n  Step 6.2: Plotting confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", color="#00d2ff")

    for ax, cm, name, cmap in zip(
        axes,
        [eval_results["svm_cm"], eval_results["rf_cm"]],
        ["SVM (RBF)", "Random Forest"],
        ["Blues", "Greens"],
    ):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, ax=ax,
            xticklabels=["Control", "Dysarthric"],
            yticklabels=["Control", "Dysarthric"],
            annot_kws={"size": 16, "fontweight": "bold"},
            linewidths=2, linecolor="#1a1a2e",
        )
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {OUTPUT_DIR / '02_confusion_matrices.png'}")
    checklist.mark_done("6.2", "Plot confusion matrices as heatmaps")

    # ── 6.3: ROC Curves ─────────────────────────────────────────────────
    print("\n  Step 6.3: Plotting ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("ROC Curves - SVM vs Random Forest",
                 fontsize=16, fontweight="bold", color="#00d2ff")

    svm_auc = eval_results["svm_auc"]
    rf_auc = eval_results["rf_auc"]

    ax.plot(eval_results["svm_fpr"], eval_results["svm_tpr"],
            color="#ff6b6b", linewidth=2.5, label=f"SVM (AUC = {svm_auc:.4f})")
    ax.plot(eval_results["rf_fpr"], eval_results["rf_tpr"],
            color="#00d2ff", linewidth=2.5, label=f"Random Forest (AUC = {rf_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#666666", linewidth=1,
            linestyle="--", label="Random Baseline")
    ax.fill_between(eval_results["svm_fpr"], eval_results["svm_tpr"],
                    alpha=0.1, color="#ff6b6b")
    ax.fill_between(eval_results["rf_fpr"], eval_results["rf_tpr"],
                    alpha=0.1, color="#00d2ff")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=11, loc="lower right", facecolor="#16213e", edgecolor="#333366")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {OUTPUT_DIR / '03_roc_curves.png'}")
    checklist.mark_done("6.3", "Plot ROC curves")

    # ── 6.4: Feature Importances ─────────────────────────────────────────
    print("\n  Step 6.4: Plotting feature importances (Top 20)...")
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Top 20 Feature Importances (Random Forest)",
                 fontsize=16, fontweight="bold", color="#00d2ff")

    top_20 = eval_results["importance_df"].head(20).sort_values("importance", ascending=True)
    bars = ax.barh(top_20["feature"], top_20["importance"],
                   color="#00d2ff", edgecolor="#333366", linewidth=0.8)

    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.cool(i / len(bars)))

    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Gini Importance", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_feature_importances.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {OUTPUT_DIR / '04_feature_importances.png'}")
    checklist.mark_done("6.4", "Plot feature importance (top 20)")

    # ── 6.5: Model Comparison Bar Chart ──────────────────────────────────
    print("\n  Step 6.5: Generating model comparison chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Model Performance Comparison",
                 fontsize=16, fontweight="bold", color="#00d2ff")

    metrics_names = ["Accuracy", "F1 (weighted)", "ROC-AUC", "CV Accuracy"]
    svm_metrics = [eval_results["svm_acc"], eval_results["svm_f1"],
                   eval_results["svm_auc"], eval_results["svm_cv_mean"]]
    rf_metrics = [eval_results["rf_acc"], eval_results["rf_f1"],
                  eval_results["rf_auc"], eval_results["rf_cv_mean"]]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, svm_metrics, width, label="SVM",
                   color="#ff6b6b", edgecolor="#333366")
    bars2 = ax.bar(x + width / 2, rf_metrics, width, label="Random Forest",
                   color="#00d2ff", edgecolor="#333366")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.legend(fontsize=11, facecolor="#16213e", edgecolor="#333366")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#ffffff")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {OUTPUT_DIR / '05_model_comparison.png'}")
    checklist.mark_done("6.5", "Generate final summary report with all metrics")


# ── Standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Phase 6 must be run after Phases 1-5. Use run_pipeline.py instead.")
