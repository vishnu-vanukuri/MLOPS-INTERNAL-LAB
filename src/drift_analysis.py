"""
drift_analysis.py
Performs automated data drift analysis using Evidently AI.
Compares reference transaction data with current production transaction stream.
Generates an interactive HTML dashboard and structured JSON summary.
"""

import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.report import Report


def run_drift_analysis(
    ref_path: str = "Section7_Q3_Online_Payment_Fraud_Reference.csv",
    cur_path: str = "Section7_Q3_Payment_Fraud_Current_Drift.csv",
    report_html: str = "reports/evidently_drift_report.html",
    report_json: str = "reports/evidently_drift_summary.json",
):
    os.makedirs(os.path.dirname(report_html), exist_ok=True)

    print("=" * 70)
    print("EVIDENTLY AI: PAYMENT TRANSACTION DATA DRIFT ANALYSIS")
    print("=" * 70)

    # 1. Load datasets
    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)

    print(f"Reference Dataset: {ref_path} (Shape: {ref_df.shape})")
    print(f"Current Dataset:   {cur_path} (Shape: {cur_df.shape})")

    # Drop identifier column
    ref_clean = ref_df.drop(columns=["Transaction_ID"], errors="ignore")
    cur_clean = cur_df.drop(columns=["Transaction_ID"], errors="ignore")

    # 2. Build & Run Evidently Report
    print("\nRunning Evidently AI DataDrift & DataQuality presets...")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref_clean, current_data=cur_clean)

    # 3. Save Artifacts
    report.save_html(report_html)
    report.save_json(report_json)
    print(f"Saved interactive HTML report: {report_html}")
    print(f"Saved JSON metrics summary:    {report_json}")

    # 4. Parse & Display Key Drift Findings
    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    drift_summary = {}
    for metric_entry in data.get("metrics", []):
        m_name = metric_entry.get("metric")
        res = metric_entry.get("result", {})

        if m_name == "DatasetDriftMetric":
            dataset_drift = res.get("dataset_drift", False)
            share_drifted = res.get("share_of_drifted_columns", 0)
            n_drifted = res.get("number_of_drifted_columns", 0)
            n_total = res.get("number_of_columns", 0)

            print("\n" + "-" * 75)
            print("DATASET-LEVEL DRIFT EVALUATION")
            print("-" * 75)
            print(f"Dataset Drift Flag:         {'DRIFT DETECTED!' if dataset_drift else 'NO DRIFT'}")
            print(f"Drifted Features Count:     {n_drifted} of {n_total} ({share_drifted:.1%})")
            print("-" * 75)

        elif m_name == "DataDriftTable":
            cols_dict = res.get("drift_by_columns", {})
            print(f"{'Feature Name':<32} | {'Drift Status':<10} | {'Stat Test':<18} | {'Score / p-val'}")
            print("-" * 75)
            for col, details in cols_dict.items():
                is_drift = details.get("drift_detected", False)
                test_name = details.get("stattest_name", "N/A")
                score = details.get("drift_score", 0.0)
                status_str = "DRIFT" if is_drift else "STABLE"

                drift_summary[col] = {
                    "drift_detected": is_drift,
                    "test": test_name,
                    "score": round(score, 6),
                }
                print(f"{col:<32} | {status_str:<10} | {test_name:<18} | {score:.6f}")
            print("-" * 75)

    return drift_summary


if __name__ == "__main__":
    run_drift_analysis()
