"""Credit-card style synthetic transaction generator with realistic fraud patterns."""
import numpy as np
import pandas as pd

CAT_MERCHANT = ["grocery", "restaurant", "fuel", "fashion", "electronics", "travel", "food_delivery", "digital", "financial", "casino"]
CAT_COUNTRY = ["US", "IN", "UK", "DE", "FR", "BR", "AU", "CA", "SG", "AE"]
CAT_DEVICE = ["mobile", "desktop", "tablet", "pos"]

MERCHANT_RISK = {
    "casino": 1.0, "electronics": 0.7, "travel": 0.55, "digital": 0.4,
    "fuel": 0.25, "fashion": 0.2, "food_delivery": 0.2, "grocery": 0.1,
    "financial": 0.1, "restaurant": 0.0,
}
MERCHANT_WEIGHT = {
    "grocery": 20, "restaurant": 22, "fuel": 10, "fashion": 8, "electronics": 7,
    "travel": 6, "food_delivery": 9, "digital": 6, "financial": 8, "casino": 4,
}
COUNTRY_WEIGHT = {"US": 40, "IN": 18, "UK": 8, "DE": 7, "FR": 6, "BR": 5, "CA": 5, "AU": 4, "AE": 4, "SG": 3}
DEVICE_WEIGHT = {"mobile": 50, "desktop": 30, "tablet": 15, "pos": 5}

HOUR_WEIGHTS = np.array([4, 2, 1, 1, 1, 2, 4, 6, 8, 8, 7, 7, 6, 6, 6, 6, 7, 8, 9, 8, 8, 7, 6, 4], dtype=float)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()


def _calibrate(logit, target_rate, n_iter=40):
    lo, hi = -30.0, 30.0
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if sig(logit + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def generate_dataset(n=45000, seed=42, target_fraud_rate=0.016):
    rng = np.random.default_rng(seed)

    merchant_w = np.array([MERCHANT_WEIGHT[m] for m in CAT_MERCHANT], dtype=float)
    country_w = np.array([COUNTRY_WEIGHT[c] for c in CAT_COUNTRY], dtype=float)
    device_w = np.array([DEVICE_WEIGHT[d] for d in CAT_DEVICE], dtype=float)

    merchant = rng.choice(CAT_MERCHANT, n, p=merchant_w / merchant_w.sum())
    country = rng.choice(CAT_COUNTRY, n, p=country_w / country_w.sum())
    device = rng.choice(CAT_DEVICE, n, p=device_w / device_w.sum())

    transaction_hour = rng.choice(24, n, p=HOUR_WEIGHTS)
    transaction_amount = np.clip(np.exp(rng.normal(4.15, 0.9, n)), 1.0, 5000.0)
    avg_amount_30d = np.clip(np.exp(rng.normal(4.15, 0.55, n)), 5.0, 3000.0)
    amount_to_avg_ratio = transaction_amount / (avg_amount_30d + 1e-9)
    account_age_days = rng.exponential(420, n).clip(1, 6000)
    previous_chargebacks = rng.poisson(0.18, n)
    tx_last_1h = rng.poisson(1.1, n)
    tx_last_24h = rng.poisson(6.0, n)

    is_night_transaction = ((transaction_hour >= 22) | (transaction_hour <= 5)).astype(int)
    is_international = (country != "US").astype(int)
    is_high_risk_merchant = (rng.random(n) < np.clip(0.02 + 0.45 * np.array([MERCHANT_RISK[m] for m in merchant]), 0.0, 0.8)).astype(int)

    logit = (
        -7.0
        + 1.1 * np.clip(amount_to_avg_ratio - 1.5, 0.0, 6.0)
        + 0.9 * np.clip(tx_last_1h, 0, 5)
        + 0.14 * np.clip(tx_last_24h, 0, 12)
        + 1.6 * previous_chargebacks
        + 1.4 * is_night_transaction
        + 1.2 * is_international
        + 1.9 * is_high_risk_merchant
        - 0.0022 * account_age_days
        + 1.8 * np.clip(np.array([MERCHANT_RISK[m] for m in merchant]), 0.0, 1.0)
        + 0.5 * is_night_transaction * is_international
        + 0.8 * np.clip(amount_to_avg_ratio - 2.0, 0.0, 4.0) * is_night_transaction
    )
    logit += np.where(device == "pos", 0.7, 0.0)
    logit += np.where(device == "mobile", 0.25, 0.0)

    offset = _calibrate(logit, target_fraud_rate)
    prob = 1.0 / (1.0 + np.exp(-(logit + offset)))
    tx_fraud = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "transaction_amount": transaction_amount.round(2),
            "avg_amount_30d": avg_amount_30d.round(2),
            "amount_to_avg_ratio": amount_to_avg_ratio.round(4),
            "account_age_days": account_age_days.round(0).astype(int),
            "previous_chargebacks": previous_chargebacks.astype(int),
            "tx_last_1h": tx_last_1h.astype(int),
            "tx_last_24h": tx_last_24h.astype(int),
            "transaction_hour": transaction_hour.astype(int),
            "is_night_transaction": is_night_transaction,
            "is_international": is_international,
            "is_high_risk_merchant": is_high_risk_merchant,
            "merchant_category": merchant,
            "country": country,
            "device_type": device,
            "TX_FRAUD": tx_fraud,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.shape)
    print("fraud rate:", df["TX_FRAUD"].mean().round(4))
    print(df.head())