"""
train.py
Trains a Machine Learning Fraud Detection classifier on the transaction dataset.
Evaluates model performance with Precision, Recall, F1, ROC-AUC, and PR-AUC.
Saves the trained pipeline, evaluation metrics, and diagnostic plots.
"""

import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    FullPreprocessingPipeline,
)


def load_and_inspect_data(csv_path: str):
    """Loads dataset and prints exploratory summary."""
    df = pd.read_csv(csv_path)
    print("=" * 60)
    print(f"Loaded dataset: {csv_path}")
    print(f"Total Rows: {len(df)}, Total Columns: {len(df.columns)}")
    print("=" * 60)

    # Missing value analysis
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    print("\n[Missing Values Summary]:")
    if len(missing_cols) == 0:
        print("  No missing values detected.")
    else:
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"  - {col}: {count} missing ({pct:.2f}%)")

    # Target class distribution
    print("\n[Target Distribution - Fraud_Label]:")
    counts = df["Fraud_Label"].value_counts()
    for label, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"  - {label}: {count} ({pct:.2f}%)")

    return df


def train_and_evaluate(csv_path: str = "Section7_Q3_Online_Payment_Fraud_Reference.csv"):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = load_and_inspect_data(csv_path)

    # Feature matrix X and target vector y
    feature_cols = [c for c in df.columns if c not in ["Transaction_ID", "Fraud_Label"]]
    X = df[feature_cols].copy()
    y = (df["Fraud_Label"] == "Fraudulent").astype(int)

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n[Split Sizes] Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Train fraud cases: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"Test fraud cases:  {y_test.sum()} ({y_test.mean():.2%})")

    # Define Full Pipeline
    preprocessor = FullPreprocessingPipeline()
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # Train model
    print("\n[Model Training] Fitting Random Forest Fraud Classifier...")
    full_pipeline.fit(X_train, y_train)

    # Cross-validation prediction on training set for robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_train_pred_prob = cross_val_predict(
        full_pipeline, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    # Predict on unseen test set
    y_test_pred_prob = full_pipeline.predict_proba(X_test)[:, 1]

    # Calculate optimal threshold based on F1-score on train CV
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    # Default to 0.5 if edge case
    if best_threshold < 0.1 or best_threshold > 0.9:
        best_threshold = 0.5

    # Predictions with standard (0.5) and optimal threshold
    y_pred_default = (y_test_pred_prob >= 0.5).astype(int)
    y_pred_tuned = (y_test_pred_prob >= best_threshold).astype(int)

    # Metrics computation
    acc = accuracy_score(y_test, y_pred_default)
    prec_fraud = precision_score(y_test, y_pred_default, zero_division=0)
    rec_fraud = recall_score(y_test, y_pred_default, zero_division=0)
    f1_fraud = f1_score(y_test, y_pred_default, zero_division=0)
    f1_macro = f1_score(y_test, y_pred_default, average="macro")
    roc_auc = roc_auc_score(y_test, y_test_pred_prob)
    pr_auc = average_precision_score(y_test, y_test_pred_prob)
    cm = confusion_matrix(y_test, y_pred_default)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (Unseen Test Set)")
    print("=" * 60)
    print(f"Accuracy:                   {acc:.4f}")
    print(f"ROC-AUC Score:              {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC): {pr_auc:.4f}")
    print(f"Fraud Precision:            {prec_fraud:.4f}")
    print(f"Fraud Recall:               {rec_fraud:.4f}")
    print(f"Fraud F1-Score:             {f1_fraud:.4f}")
    print(f"Macro F1-Score:             {f1_macro:.4f}")
    print(f"Optimal Decision Threshold: {best_threshold:.4f}")

    print("\n[Detailed Classification Report]:")
    print(
        classification_report(
            y_test,
            y_pred_default,
            target_names=["Legitimate", "Fraudulent"],
            digits=4,
        )
    )

    print("\n[Confusion Matrix]:")
    print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")

    # Save metrics to JSON
    metrics = {
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "fraud_support_test": int(y_test.sum()),
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_fraud": float(prec_fraud),
        "recall_fraud": float(rec_fraud),
        "f1_fraud": float(f1_fraud),
        "f1_macro": float(f1_macro),
        "decision_threshold": float(best_threshold),
        "confusion_matrix": {
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        },
    }

    with open("models/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("\nSaved evaluation metrics to models/evaluation_metrics.json")

    # Save serialized model bundle
    joblib.dump(full_pipeline, "models/fraud_model_pipeline.joblib")
    print("Saved trained model pipeline to models/fraud_model_pipeline.joblib")

    # Feature names
    feature_names = full_pipeline.named_steps["preprocessor"].feature_names_
    with open("models/feature_names.json", "w") as f:
        json.dump(list(feature_names), f, indent=4)

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Fraud Detection", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Legitimate", "Fraudulent"])
    plt.yticks(tick_marks, ["Legitimate", "Fraudulent"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=200)
    plt.close()

    # Plot 2: ROC and PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    axes[0].plot(
        fpr,
        tpr,
        color="#2b5c8f",
        lw=2.5,
        label=f"Random Forest (AUC = {roc_auc:.3f})",
    )
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # PR Curve
    pr_p, pr_r, _ = precision_recall_curve(y_test, y_test_pred_prob)
    axes[1].plot(
        pr_r,
        pr_p,
        color="#c84b31",
        lw=2.5,
        label=f"PR Curve (AP = {pr_auc:.3f})",
    )
    axes[1].axhline(
        y=y_test.mean(),
        color="gray",
        linestyle="--",
        label=f"Baseline ({y_test.mean():.2%})",
    )
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/roc_pr_curve.png", dpi=200)
    plt.close()

    # Plot 3: Feature Importances
    rf_model = full_pipeline.named_steps["classifier"]
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.barh(
        range(len(sorted_idx)),
        importances[sorted_idx][::-1],
        align="center",
        color="#1f77b4",
    )
    plt.yticks(
        range(len(sorted_idx)),
        [feature_names[i] for i in sorted_idx][::-1],
        fontsize=10,
    )
    plt.xlabel("Gini Feature Importance", fontsize=11)
    plt.title(
        "Top 15 Predictive Features - Fraud Detection Model",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=200)
    plt.close()

    print("Saved evaluation plots to reports/ directory:")
    print("  - reports/confusion_matrix.png")
    print("  - reports/roc_pr_curve.png")
    print("  - reports/feature_importance.png")
    return full_pipeline, metrics


if __name__ == "__main__":
    train_and_evaluate()
