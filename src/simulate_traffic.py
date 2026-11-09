"""
simulate_traffic.py
Simulates real-world traffic patterns against the Fraud Detection API:
  Phase 1: Normal Baseline Traffic (Reference data distribution)
  Phase 2: Drifted Traffic Pattern (Elevated amounts, higher frequency, Card shifts)
  Phase 3: Error & Edge Case Injection (Testing logging and error metrics)
Demonstrates real-time monitoring and metric changes.
"""

import json
import os
import sys
import time
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests
from fastapi.testclient import TestClient

from src.app import app


class TrafficSimulator:
    def __init__(self, base_url: str = None):
        """
        If base_url is provided, uses HTTP requests against running server.
        Otherwise, uses FastAPI TestClient in-memory for seamless standalone testing.
        """
        self.base_url = base_url
        self.client = None
        if not base_url:
            self.client = TestClient(app)
        self.results = {
            "baseline": {"requests": 0, "legitimate": 0, "fraudulent": 0, "latencies_ms": []},
            "drifted":  {"requests": 0, "legitimate": 0, "fraudulent": 0, "latencies_ms": []},
            "errors":   {"requests": 0, "error_responses": 0, "details": []},
        }

    def _send_request(self, payload: dict) -> dict:
        if self.base_url:
            url = f"{self.base_url}/predict"
            resp = requests.post(url, json=payload, timeout=5)
            return {"status_code": resp.status_code, "data": resp.json() if resp.status_code == 200 else resp.text}
        else:
            resp = self.client.post("/predict", json=payload)
            return {"status_code": resp.status_code, "data": resp.json() if resp.status_code == 200 else resp.text}

    def _get_metrics(self) -> str:
        if self.base_url:
            resp = requests.get(f"{self.base_url}/metrics", timeout=5)
            return resp.text
        else:
            resp = self.client.get("/metrics")
            return resp.text

    def simulate_baseline_traffic(self, n_samples: int = 50):
        print(f"\n>>> [Phase 1] Simulating Baseline Normal Traffic ({n_samples} requests)...")
        ref_df = pd.read_csv("Section7_Q3_Online_Payment_Fraud_Reference.csv")
        sample_df = ref_df.sample(n=min(n_samples, len(ref_df)), random_state=42)

        for _, row in sample_df.iterrows():
            payload = {
                "Transaction_ID": str(row["Transaction_ID"]),
                "Transaction_Amount_INR": float(row["Transaction_Amount_INR"]) if pd.notnull(row["Transaction_Amount_INR"]) else 500.0,
                "Transaction_Hour": int(row["Transaction_Hour"]),
                "Payment_Method": str(row["Payment_Method"]) if pd.notnull(row["Payment_Method"]) else "UPI",
                "Transaction_Channel": str(row["Transaction_Channel"]),
                "Device_Status": str(row["Device_Status"]) if pd.notnull(row["Device_Status"]) else "Known",
                "Location_Type": str(row["Location_Type"]),
                "Transactions_Last_Hour": float(row["Transactions_Last_Hour"]) if pd.notnull(row["Transactions_Last_Hour"]) else 1.0,
                "Account_Age_Days": int(row["Account_Age_Days"]),
                "Fraud_Risk_Signal": float(row["Fraud_Risk_Signal"]) if pd.notnull(row["Fraud_Risk_Signal"]) else 50.0,
                "Failed_OTP_Attempts": int(row["Failed_OTP_Attempts"]),
                "Previous_Disputes": int(row["Previous_Disputes"]),
                "Minutes_Since_Last_Transaction": int(row["Minutes_Since_Last_Transaction"]),
            }

            resp = self._send_request(payload)
            if resp["status_code"] == 200:
                data = resp["data"]
                self.results["baseline"]["requests"] += 1
                if data["prediction"] == "Fraudulent":
                    self.results["baseline"]["fraudulent"] += 1
                else:
                    self.results["baseline"]["legitimate"] += 1
                self.results["baseline"]["latencies_ms"].append(data["processing_time_ms"])

        avg_lat = np.mean(self.results["baseline"]["latencies_ms"]) if self.results["baseline"]["latencies_ms"] else 0
        print(f"    Completed: {self.results['baseline']['requests']} requests")
        print(f"    Legitimate: {self.results['baseline']['legitimate']}, Fraudulent: {self.results['baseline']['fraudulent']}")
        print(f"    Mean Latency: {avg_lat:.2f} ms")

    def simulate_drifted_traffic(self, n_samples: int = 50):
        print(f"\n>>> [Phase 2] Simulating Drifted Traffic (Higher amounts, Card surge, International) ({n_samples} requests)...")
        drift_df = pd.read_csv("Section7_Q3_Payment_Fraud_Current_Drift.csv")
        sample_df = drift_df.sample(n=min(n_samples, len(drift_df)), random_state=99)

        for _, row in sample_df.iterrows():
            payload = {
                "Transaction_ID": str(row["Transaction_ID"]),
                "Transaction_Amount_INR": float(row["Transaction_Amount_INR"]) if pd.notnull(row["Transaction_Amount_INR"]) else 2500.0,
                "Transaction_Hour": int(row["Transaction_Hour"]),
                "Payment_Method": str(row["Payment_Method"]) if pd.notnull(row["Payment_Method"]) else "Card",
                "Transaction_Channel": str(row["Transaction_Channel"]),
                "Device_Status": str(row["Device_Status"]) if pd.notnull(row["Device_Status"]) else "New",
                "Location_Type": str(row["Location_Type"]),
                "Transactions_Last_Hour": float(row["Transactions_Last_Hour"]) if pd.notnull(row["Transactions_Last_Hour"]) else 4.0,
                "Account_Age_Days": int(row["Account_Age_Days"]),
                "Fraud_Risk_Signal": float(row["Fraud_Risk_Signal"]) if pd.notnull(row["Fraud_Risk_Signal"]) else 65.0,
                "Failed_OTP_Attempts": int(row["Failed_OTP_Attempts"]),
                "Previous_Disputes": int(row["Previous_Disputes"]),
                "Minutes_Since_Last_Transaction": int(row["Minutes_Since_Last_Transaction"]),
            }

            resp = self._send_request(payload)
            if resp["status_code"] == 200:
                data = resp["data"]
                self.results["drifted"]["requests"] += 1
                if data["prediction"] == "Fraudulent":
                    self.results["drifted"]["fraudulent"] += 1
                else:
                    self.results["drifted"]["legitimate"] += 1
                self.results["drifted"]["latencies_ms"].append(data["processing_time_ms"])

        avg_lat = np.mean(self.results["drifted"]["latencies_ms"]) if self.results["drifted"]["latencies_ms"] else 0
        print(f"    Completed: {self.results['drifted']['requests']} requests")
        print(f"    Legitimate: {self.results['drifted']['legitimate']}, Fraudulent: {self.results['drifted']['fraudulent']}")
        print(f"    Mean Latency: {avg_lat:.2f} ms")

    def simulate_error_traffic(self, n_samples: int = 10):
        print(f"\n>>> [Phase 3] Simulating Error Cases & Malformed Payloads ({n_samples} requests)...")
        error_cases = [
            {"Transaction_Amount_INR": -500.0, "Transaction_Hour": 12},  # Negative amount
            {"Transaction_Amount_INR": 100.0, "Transaction_Hour": 28},   # Invalid hour > 23
            {"Transaction_Amount_INR": 0.0},                              # Amount = 0
            {"Payment_Method": "InvalidCoin"},                            # Missing required fields
            {},                                                          # Completely empty payload
        ]

        for i, case in enumerate(error_cases):
            resp = self._send_request(case)
            self.results["errors"]["requests"] += 1
            if resp["status_code"] != 200:
                self.results["errors"]["error_responses"] += 1
                self.results["errors"]["details"].append({
                    "case_id": i + 1,
                    "status_code": resp["status_code"],
                    "response": resp["data"][:120] if isinstance(resp["data"], str) else resp["data"],
                })

        print(f"    Completed: {self.results['errors']['requests']} error injection requests")
        print(f"    Intercepted Errors: {self.results['errors']['error_responses']}")

    def print_monitoring_summary(self):
        print("\n" + "=" * 70)
        print("SIMULATION & MONITORING SUMMARY COMPARISON")
        print("=" * 70)
        b = self.results["baseline"]
        d = self.results["drifted"]
        e = self.results["errors"]

        b_lat_mean = np.mean(b["latencies_ms"]) if b["latencies_ms"] else 0
        b_lat_p95 = np.percentile(b["latencies_ms"], 95) if b["latencies_ms"] else 0
        d_lat_mean = np.mean(d["latencies_ms"]) if d["latencies_ms"] else 0
        d_lat_p95 = np.percentile(d["latencies_ms"], 95) if d["latencies_ms"] else 0

        print(f"{'Metric':<32} | {'Baseline (Normal)':<18} | {'Drifted Stream'}")
        print("-" * 70)
        print(f"{'Total Requests':<32} | {b['requests']:<18} | {d['requests']}")
        print(f"{'Legitimate Count':<32} | {b['legitimate']:<18} | {d['legitimate']}")
        print(f"{'Fraudulent Count':<32} | {b['fraudulent']:<18} | {d['fraudulent']}")
        b_fraud_rate = (b['fraudulent']/b['requests']*100) if b['requests'] else 0
        d_fraud_rate = (d['fraudulent']/d['requests']*100) if d['requests'] else 0
        print(f"{'Flagged Fraud Rate (%)':<32} | {b_fraud_rate:<17.1f}% | {d_fraud_rate:.1f}%")
        print(f"{'Mean Latency (ms)':<32} | {b_lat_mean:<18.2f} | {d_lat_mean:.2f}")
        print(f"{'P95 Latency (ms)':<32} | {b_lat_p95:<18.2f} | {d_lat_p95:.2f}")
        print(f"{'Errors Handled':<32} | {'N/A':<18} | {e['error_responses']} errors logged")
        print("-" * 70)

        # Scrape and show sample Prometheus lines
        metrics_raw = self._get_metrics()
        print("\n[Prometheus Live Scrape Sample]:")
        for line in metrics_raw.splitlines():
            if any(k in line for k in ["fraud_predictions_total", "fraud_http_requests_total", "fraud_errors_total"]) and not line.startswith("#"):
                print("  ", line)

        # Save simulation summary
        os.makedirs("reports", exist_ok=True)
        summary_to_save = {
            "baseline": {
                "requests": b["requests"],
                "legitimate": b["legitimate"],
                "fraudulent": b["fraudulent"],
                "fraud_rate_pct": round(b_fraud_rate, 2),
                "mean_latency_ms": round(float(b_lat_mean), 2),
                "p95_latency_ms": round(float(b_lat_p95), 2),
            },
            "drifted": {
                "requests": d["requests"],
                "legitimate": d["legitimate"],
                "fraudulent": d["fraudulent"],
                "fraud_rate_pct": round(d_fraud_rate, 2),
                "mean_latency_ms": round(float(d_lat_mean), 2),
                "p95_latency_ms": round(float(d_lat_p95), 2),
            },
            "errors": {
                "tested": e["requests"],
                "handled": e["error_responses"],
            },
        }
        with open("reports/simulation_results.json", "w") as f:
            json.dump(summary_to_save, f, indent=4)
        print("\nSimulation results persisted to reports/simulation_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=None, help="Base URL of live FastAPI server (e.g., http://localhost:8000)")
    parser.add_argument("--n-samples", type=int, default=60, help="Number of samples per phase")
    args = parser.parse_args()

    simulator = TrafficSimulator(base_url=args.url)
    simulator.simulate_baseline_traffic(n_samples=args.n_samples)
    simulator.simulate_drifted_traffic(n_samples=args.n_samples)
    simulator.simulate_error_traffic(n_samples=10)
    simulator.print_monitoring_summary()
