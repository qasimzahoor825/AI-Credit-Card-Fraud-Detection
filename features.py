"""Feature builder shared by the training pipeline and the app (inference only)."""
import numpy as np
import pandas as pd

RAW_FIELDS = [
    "transaction_amount",
    "avg_amount_30d",
    "account_age_days",
    "previous_chargebacks",
    "tx_last_1h",
    "tx_last_24h",
    "transaction_hour",
    "merchant_category",
    "country",
    "device_type",
    "is_international",
    "is_high_risk_merchant",
]

NUM_COLS = [
    "transaction_amount",
    "avg_amount_30d",
    "amount_to_avg_ratio",
    "account_age_days",
    "previous_chargebacks",
    "tx_last_1h",
    "tx_last_24h",
    "transaction_hour",
    "is_night_transaction",
    "is_international",
    "is_high_risk_merchant",
]

CAT_COLS = ["merchant_category", "country", "device_type"]

INPUT_COLUMNS = NUM_COLS + CAT_COLS

CAT_MERCHANT = ["grocery", "restaurant", "fuel", "fashion", "electronics", "travel", "food_delivery", "digital", "financial", "casino"]
CAT_COUNTRY = ["US", "IN", "UK", "DE", "FR", "BR", "AU", "CA", "SG", "AE"]
CAT_DEVICE = ["mobile", "desktop", "tablet", "pos"]


def enrich(df):
    d = df.copy()
    d["amount_to_avg_ratio"] = d["transaction_amount"] / (d["avg_amount_30d"].replace(0, np.nan).fillna(1.0) + 1e-9)
    d["is_night_transaction"] = ((d["transaction_hour"] >= 22) | (d["transaction_hour"] <= 5)).astype(int)
    d["is_international"] = d["is_international"].astype(int)
    d["is_high_risk_merchant"] = d["is_high_risk_merchant"].astype(int)
    for col in NUM_COLS[:8]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d[INPUT_COLUMNS]


def row_to_df(values):
    return pd.DataFrame([values])[RAW_FIELDS]


def clean_batch(df):
    d = df.copy()
    for col in ["is_international", "is_high_risk_merchant"]:
        if col in d.columns:
            d[col] = d[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0}).fillna(d[col])
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)
    return d