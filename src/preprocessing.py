"""
preprocessing.py
Preprocesses transaction data, handles missing values, and implements feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Raw feature definitions
CATEGORICAL_FEATURES = [
    "Payment_Method",
    "Transaction_Channel",
    "Device_Status",
    "Location_Type",
]

NUMERICAL_FEATURES = [
    "Transaction_Amount_INR",
    "Transaction_Hour",
    "Transactions_Last_Hour",
    "Account_Age_Days",
    "Fraud_Risk_Signal",
    "Failed_OTP_Attempts",
    "Previous_Disputes",
    "Minutes_Since_Last_Transaction",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer risk and domain features:
    - Amount_to_Account_Age_Ratio
    - Is_Night_Transaction (Transactions between 23:00 and 05:00)
    - Dispute_Rate (Disputes normalized by account tenure)
    - Velocity_Per_Minute (Transaction velocity normalized by elapsed minutes)
    - Otp_Dispute_Risk_Composite (Composite risk multiplier)
    - Log_Amount (Log transformed transaction amount)
    """

    def __init__(self):
        self.engineered_feature_names_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expects a DataFrame
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # 1. Amount to Account Age Ratio
        df["Amount_to_Account_Age_Ratio"] = df["Transaction_Amount_INR"] / (
            df["Account_Age_Days"] + 1.0
        )

        # 2. Night Transaction flag (23:00 to 05:00)
        df["Is_Night_Transaction"] = df["Transaction_Hour"].apply(
            lambda h: 1.0 if (h >= 23 or h <= 5) else 0.0
        )

        # 3. Dispute Rate per year of account age
        df["Dispute_Rate"] = df["Previous_Disputes"] / (
            (df["Account_Age_Days"] / 365.0) + 1.0
        )

        # 4. Velocity Per Minute
        df["Velocity_Per_Minute"] = df["Transactions_Last_Hour"] / (
            df["Minutes_Since_Last_Transaction"] + 1.0
        )

        # 5. Composite Risk Signal
        df["Otp_Dispute_Risk_Composite"] = (
            (df["Failed_OTP_Attempts"] + 1.0)
            * (df["Previous_Disputes"] + 1.0)
            * (df["Fraud_Risk_Signal"] / 100.0)
        )

        # 6. Log of Amount to handle right-skewed amounts
        df["Log_Amount"] = np.log1p(np.maximum(df["Transaction_Amount_INR"], 0))

        return df


def get_all_numerical_cols():
    return NUMERICAL_FEATURES + [
        "Amount_to_Account_Age_Ratio",
        "Is_Night_Transaction",
        "Dispute_Rate",
        "Velocity_Per_Minute",
        "Otp_Dispute_Risk_Composite",
        "Log_Amount",
    ]


def build_preprocessor_pipeline():
    """
    Constructs an end-to-end preprocessing pipeline:
    1. Imputation (Median for numerical, Most Frequent for categorical)
    2. Feature Engineering
    3. Scaling (StandardScaler) and One-Hot Encoding (OHE)
    """
    # Imputers before feature engineering
    # We do simple imputation first on raw columns
    num_imputer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # We will wrap feature engineering into a full workflow
    return num_imputer, cat_pipeline


class FullPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end transformer that handles missing values, feature engineering,
    standard scaling for numericals, and one-hot encoding for categoricals.
    """

    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names_ = []

    def fit(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # 1. Impute raw numericals & categoricals
        num_part = pd.DataFrame(
            self.num_imputer.fit_transform(df[NUMERICAL_FEATURES]),
            columns=NUMERICAL_FEATURES,
            index=df.index,
        )
        cat_part = pd.DataFrame(
            self.cat_imputer.fit_transform(df[CATEGORICAL_FEATURES]),
            columns=CATEGORICAL_FEATURES,
            index=df.index,
        )

        combined = pd.concat([num_part, cat_part], axis=1)

        # 2. Feature engineering on imputed data
        engineered = self.feature_engineer.transform(combined)

        all_num_cols = get_all_numerical_cols()
        self.scaler.fit(engineered[all_num_cols])
        self.ohe.fit(engineered[CATEGORICAL_FEATURES])

        # Track output column names
        ohe_cols = list(self.ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names_ = all_num_cols + ohe_cols
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Ensure all required raw columns exist
        for col in NUMERICAL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        num_part = pd.DataFrame(
            self.num_imputer.transform(df[NUMERICAL_FEATURES]),
            columns=NUMERICAL_FEATURES,
            index=df.index,
        )
        cat_part = pd.DataFrame(
            self.cat_imputer.transform(df[CATEGORICAL_FEATURES]),
            columns=CATEGORICAL_FEATURES,
            index=df.index,
        )

        combined = pd.concat([num_part, cat_part], axis=1)
        engineered = self.feature_engineer.transform(combined)

        all_num_cols = get_all_numerical_cols()
        scaled_num = self.scaler.transform(engineered[all_num_cols])
        encoded_cat = self.ohe.transform(engineered[CATEGORICAL_FEATURES])

        return np.hstack([scaled_num, encoded_cat])

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
