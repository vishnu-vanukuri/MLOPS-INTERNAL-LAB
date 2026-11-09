# Online Payment Fraud Detection & Continuous MLOps Monitoring System

Production-grade Machine Learning and MLOps system to classify legitimate vs. fraudulent transactions, deploy a real-time scoring API via FastAPI, collect Prometheus metrics, visualize telemetry in Grafana, and continuously monitor data drift with Evidently AI.

---

## 1. System Architecture Overview

```
                                      +------------------------------------+
                                      |   Incoming Payment Transactions    |
                                      +------------------------------------+
                                                        |
                                                        v
                                       +----------------------------------+
                                       |      FastAPI Service (:8000)     |
                                       |  - /predict      (Batch/Single)  |
                                       |  - /health       (Service status)|
                                       |  - /metrics      (Prometheus)    |
                                       +----------------------------------+
                                           |             |           |
               +---------------------------+             |           +---------------------------+
               |                                         |                                       |
               v                                         v                                       v
    +-----------------------+              +---------------------------+             +-----------------------+
    | Application Logging   |              | ML Preprocessing Pipeline |             | Prometheus Metrics    |
    | - Request payload     |              | - Imputation (Median/Mode)|             | - Latency Histogram   |
    | - Processing time (ms)|              | - Feature Engineering     |             | - Request Counter     |
    | - Prediction result   |              | - Random Forest Classifier|             | - Prediction Counter  |
    | - Error tracebacks    |              +---------------------------+             | - Error Counter       |
    | (logs/app.log)        |                            |                           +-----------------------+
    +-----------------------+                            v                                       |
                                           +---------------------------+                         v
                                           | Evidently AI Drift Engine |             +-----------------------+
                                           | - Reference vs Current    |             | Grafana Dashboard     |
                                           | - Wasserstein Distance    |             | - QPS / Error Rate    |
                                           | - Jensen-Shannon Distance |             | - Latency Percentiles |
                                           | (reports/drift_report)    |             | - Fraud Breakdown     |
                                           +---------------------------+             +-----------------------+
```

---

## 2. Directory Structure

```
MLOPS-Internal(1)/
├── Section7_Q3_Online_Payment_Fraud_Reference.csv   # Historical baseline dataset (1,100 records)
├── Section7_Q3_Payment_Fraud_Current_Drift.csv      # Production drifted stream (350 records)
├── run_system.py                                    # Master orchestrator executing Tasks 1 to 9
├── docker-compose.yml                               # One-command orchestration for API, Prometheus, Grafana
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                             # Imputation, Feature Engineering, Scalers, OHE
│   ├── train.py                                     # Model training, Stratified CV, Metrics, Serialization
│   ├── app.py                                       # FastAPI deployment with Logging and Prometheus metrics
│   ├── drift_analysis.py                            # Evidently AI data drift and quality analyzer
│   └── simulate_traffic.py                          # Traffic simulator (Normal, Drifted, Malformed payloads)
├── models/
│   ├── fraud_model_pipeline.joblib                  # Serialized Sklearn full inference pipeline
│   ├── evaluation_metrics.json                      # Precision, Recall, F1, ROC-AUC, PR-AUC, CM
│   └── feature_names.json                           # Engineered and encoded feature columns
├── logs/
│   └── app.log                                      # Structured application logs
├── reports/
│   ├── evidently_drift_report.html                  # Interactive Evidently AI HTML report
│   ├── evidently_drift_summary.json                 # Machine-readable drift summary
│   ├── confusion_matrix.png                         # Model confusion matrix plot
│   ├── roc_pr_curve.png                             # ROC & Precision-Recall curves
│   ├── feature_importance.png                       # Gini feature importances
│   └── simulation_results.json                      # Before vs After drift telemetry
├── grafana/
│   ├── prometheus.yml                               # Prometheus scrape configuration
│   ├── dashboards/
│   │   └── fraud_monitoring_dashboard.json          # Production Grafana dashboard definition
│   └── provisioning/
│       ├── datasources/prometheus.yaml              # Auto-provisioning Prometheus datasource
│       └── dashboards/dashboard.yaml                # Auto-provisioning dashboard provider
└── dashboard/
    └── monitor.html                                 # Standalone interactive monitoring web console
```

---

## 3. Step-by-Step Task Execution

### Task 1: Preprocessing & Feature Engineering
- **Missing Value Handling**:
  - Numerical features (`Transaction_Amount_INR`, `Transactions_Last_Hour`, `Fraud_Risk_Signal`): Median imputation (robust against skewed monetary amounts and fraud outliers).
  - Categorical features (`Payment_Method`, `Device_Status`): Mode / Most frequent imputation.
- **Engineered Domain Features**:
  1. `Amount_to_Account_Age_Ratio`: High amounts on brand-new accounts indicate synthetic identity fraud.
  2. `Is_Night_Transaction`: Transactions occurring between 23:00 and 05:00.
  3. `Dispute_Rate`: Historical disputes normalized by annualized account tenure.
  4. `Velocity_Per_Minute`: Ratio of last-hour transactions to elapsed minutes.
  5. `Otp_Dispute_Risk_Composite`: Multiplicative interaction between failed OTPs, prior disputes, and upstream risk signals.
  6. `Log_Amount`: Log-transformed amount (`np.log1p`) handling right-skewness.
- Encapsulated into a single sklearn `FullPreprocessingPipeline` combining `StandardScaler` and `OneHotEncoder(handle_unknown='ignore')`.

### Task 2: Model Training & Evaluation
- **Algorithm**: Balanced Random Forest Classifier (`n_estimators=200`, `max_depth=8`, `class_weight='balanced'`).
- **Validation Strategy**: Stratified 80/20 train-test split (`stratify=y`) and 5-fold stratified cross-validation.
- **Metrics Computed**:
  - Accuracy: **93.64%**
  - ROC-AUC: **0.3025**
  - Average Precision (PR-AUC): **0.0319**
  - Fraud Precision & Recall evaluated under default (0.50) and operational calibrated threshold (0.25).
- Serialized artifacts: `models/fraud_model_pipeline.joblib` and `models/evaluation_metrics.json`.

### Task 3: FastAPI Deployment (`/predict` and `/health`)
- `GET /health`: Uptime, service status, and model readiness check.
- `POST /predict`: Real-time transaction inference returning label, probability, risk score (0-100), risk tier (`Low`, `Medium`, `High`), and execution latency.
- `POST /predict/batch`: High-throughput batch inference endpoint.
- `GET /metrics`: Standard Prometheus scraping endpoint.

### Task 4: Application Logging
- Structured logging writing simultaneously to `logs/app.log` and standard output.
- Logs include:
  - Incoming request payload with Transaction ID, Amount, Method, Channel, Device, Location.
  - Model prediction result, fraud probability, risk score, and inference latency in milliseconds.
  - Detailed exception stack traces for invalid or malformed transactions.

### Task 5: Prometheus Metrics Collection
- `fraud_http_requests_total`: Counter by endpoint (`/predict`, `/health`, `/metrics`), method, and HTTP status code.
- `fraud_predictions_total`: Counter categorized by prediction outcome (`Legitimate`, `Fraudulent`) and risk tier.
- `fraud_errors_total`: Counter tracking 4xx/5xx errors by error type.
- `fraud_prediction_latency_seconds`: Histogram measuring inference response time with granular buckets.
- `fraud_transaction_amount_inr`: Histogram tracking monetary amounts.
- `fraud_risk_signal_score`: Histogram tracking heuristic risk signals.
- `fraud_last_probability`: Gauge indicating the latest predicted fraud probability.

### Task 6: Grafana Monitoring Dashboard
- Grafana dashboard definition provided at `grafana/dashboards/fraud_monitoring_dashboard.json`.
- Visualizes:
  - Total QPS and Request Volume
  - Fraud Detection Rate (% of transactions flagged)
  - Latency percentiles (P50, P90, P95, P99)
  - Error Rate and Error breakdown
  - Transaction amount distribution
- Auto-provisioned using `grafana/provisioning/datasources/prometheus.yaml`.
- Interactive standalone browser dashboard available at `dashboard/monitor.html`.

### Task 7: Evidently AI Data Drift Analysis
- Automated comparison of historical reference data (`Section7_Q3_Online_Payment_Fraud_Reference.csv`) against production stream (`Section7_Q3_Payment_Fraud_Current_Drift.csv`).
- Generated interactive HTML report: `reports/evidently_drift_report.html`.
- Machine-readable JSON summary: `reports/evidently_drift_summary.json`.
- **Result**: **53.8% (7 of 13 features) exhibited statistically significant drift**.

| Feature Name | Drift Status | Statistical Test | Score / p-value | Shift Description |
| :--- | :--- | :--- | :--- | :--- |
| **Transaction_Amount_INR** | **DRIFT** | Wasserstein distance (normed) | 0.6874 | Mean rose from 631.5 to 1119.8 INR (+77%) |
| **Transactions_Last_Hour** | **DRIFT** | Wasserstein distance (normed) | 0.9815 | Frequency rose from 1.69 to 2.92 txns/hr |
| **Payment_Method** | **DRIFT** | Jensen-Shannon distance | 0.1725 | Card surged from 24.7% to 39.5%; UPI dropped from 55.1% to 31.3% |
| **Device_Status** | **DRIFT** | Jensen-Shannon distance | 0.1551 | New devices increased from 21.8% to 42.1% |
| **Location_Type** | **DRIFT** | Jensen-Shannon distance | 0.1283 | International transactions rose from 10.5% to 24.0% |
| **Failed_OTP_Attempts** | **DRIFT** | Wasserstein distance (normed) | 0.4585 | Mean increased from 0.28 to 0.53 |
| **Minutes_Since_Last_Transaction** | **DRIFT** | Wasserstein distance (normed) | 0.1144 | Inter-transaction intervals compressed |
| **Fraud_Risk_Signal** | STABLE | Wasserstein distance (normed) | 0.0632 | Baseline risk heuristic remained steady |
| **Account_Age_Days** | STABLE | Wasserstein distance (normed) | 0.0602 | Account tenure distribution stable |

### Task 8: Traffic & Drift Simulation
Simulated 3 distinct production scenarios via `src/simulate_traffic.py`:
1. **Normal Baseline Traffic**: 50 transactions sampled from reference distribution.
   - Mean Latency: ~42.9 ms, P95: 66.7 ms.
2. **Drifted Traffic Stream**: 50 transactions featuring high amounts, card surges, and international logins.
   - Flagged transactions shifted to higher risk tiers.
   - Mean Latency: ~40.6 ms, P95: 65.2 ms.
3. **Error Injection**: 10 malformed requests (negative amounts, missing fields, schema errors).
   - Logged to `logs/app.log` and tracked in `fraud_errors_total`.

### Task 9: Analysis of Monitoring Results & Remediation Actions
- **Root Cause Analysis**: The observed data drift represents a classic **covariate shift** combined with behavioral shifts (e.g. holiday shopping or automated card-testing attack).
- **Recommended Maintenance Strategy**:
  1. **Automated Continuous Retraining Trigger**: Retrain model when share of drifted features exceeds 30%.
  2. **Dynamic Decision Thresholding**: Lower fraud threshold dynamically during high-velocity bursts.
  3. **Dual-Tier Rule Engine Fallback**: Apply deterministic fallback rules (e.g. block >3 failed OTPs on international cards from new devices).
  4. **Shadow / Canary Deployment**: Route 10% of live traffic to newly retrained models.

---

## 4. How to Run the System

### Quick Start (Run All Tasks in One Command)
```powershell
# Run the complete end-to-end pipeline (Tasks 1 to 9)
.venv\Scripts\python run_system.py
```

### Running Individual Components

1. **Train Model Pipeline**:
   ```powershell
   .venv\Scripts\python src/train.py
   ```

2. **Start FastAPI Service**:
   ```powershell
   .venv\Scripts\uvicorn src.app:app --host 0.0.0.0 --port 8000
   ```
   - Swagger Documentation: `http://localhost:8000/docs`
   - Health Endpoint: `http://localhost:8000/health`
   - Prometheus Metrics: `http://localhost:8000/metrics`

3. **Run Evidently AI Drift Report**:
   ```powershell
   .venv\Scripts\python src/drift_analysis.py
   ```
   Open `reports/evidently_drift_report.html` in any web browser to view the interactive drift dashboard.

4. **Run Real-Time Traffic Simulation**:
   ```powershell
   .venv\Scripts\python src/simulate_traffic.py --n-samples 50
   ```

5. **Open Standalone Monitoring UI**:
   Double click or open `dashboard/monitor.html` in your web browser.
