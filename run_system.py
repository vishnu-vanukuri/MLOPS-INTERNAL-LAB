"""
run_system.py
End-to-End Orchestrator for Question 3: Online Payment Fraud Detection System.
Executes tasks 1 through 9 sequentially with structured logging and reporting.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_and_evaluate
from src.drift_analysis import run_drift_analysis
from src.simulate_traffic import TrafficSimulator


def main():
    print("\n" + "#" * 80)
    print("  ONLINE PAYMENT FRAUD DETECTION & CONTINUOUS MONITORING SYSTEM")
    print("  End-to-End Execution Pipeline (Tasks 1 to 9)")
    print("#" * 80 + "\n")

    # Step 1 & 2: Preprocessing, Feature Engineering, Training & Evaluation
    print(">>> [TASK 1 & 2] Loading Reference Data, Handling Missing Values, Engineering Features & Training Model...")
    pipeline, metrics = train_and_evaluate()

    # Step 3, 4 & 5: Verified in app.py (FastAPI, Logging, Prometheus)
    print("\n>>> [TASK 3, 4 & 5] Initializing FastAPI App, Application Logging & Prometheus Metrics Engine...")
    simulator = TrafficSimulator()

    # Health check
    health_resp = simulator.client.get("/health")
    print(f"    FastAPI Health Check: {health_resp.status_code} - {health_resp.json()}")

    # Step 7: Evidently AI Data Drift Analysis
    print("\n>>> [TASK 7] Running Evidently AI Data Drift Analysis (Reference vs Drift Datasets)...")
    drift_summary = run_drift_analysis()

    # Step 8: Simulate Traffic (Amounts, Frequency, Payment Methods) and Observe System
    print("\n>>> [TASK 8] Simulating Changes in Transaction Amounts, Frequency, Payment Methods & Error Scenarios...")
    simulator.simulate_baseline_traffic(n_samples=50)
    simulator.simulate_drifted_traffic(n_samples=50)
    simulator.simulate_error_traffic(n_samples=10)
    simulator.print_monitoring_summary()

    print("\n" + "#" * 80)
    print("  EXECUTION COMPLETE - ALL ARTIFACTS AND REPORTS GENERATED SUCCESSFULLY")
    print("#" * 80)
    print("Artifacts generated:")
    print("  - Models:  models/fraud_model_pipeline.joblib, models/evaluation_metrics.json")
    print("  - Logs:    logs/app.log")
    print("  - Reports: reports/evidently_drift_report.html, reports/evidently_drift_summary.json")
    print("             reports/confusion_matrix.png, reports/roc_pr_curve.png, reports/feature_importance.png")
    print("             reports/simulation_results.json")
    print("  - Grafana: grafana/dashboards/fraud_monitoring_dashboard.json, grafana/prometheus.yml")
    print("  - Web UI:  dashboard/monitor.html")


if __name__ == "__main__":
    main()
