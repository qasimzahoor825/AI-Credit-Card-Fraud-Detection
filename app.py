"""AI-Powered Credit Card Fraud Detection - Streamlit dashboard.

Inference-only: loads pre-trained models + preprocessor. No retraining here.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import features
import train_model
MODEL_DIR = os.path.join(HERE, "models")
DATA_DIR = os.path.join(HERE, "data")
DATA_CSV = os.path.join(DATA_DIR, "credit_fraud_data.csv")
SAMPLE_CSV = os.path.join(DATA_DIR, "sample_transactions.csv")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


@st.cache_resource
def get_artifacts():
    if not train_model.model_exists():
        with st.spinner("First run: training the models, please wait..."):
            train_model.train()
    pre = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
    models = {
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib")),
        "XGBoost": joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib")),
        "Stacking Ensemble": joblib.load(os.path.join(MODEL_DIR, "stack_model.joblib")),
    }
    metrics = json.load(open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8"))
    roc_data = json.load(open(os.path.join(MODEL_DIR, "roc_data.json"), encoding="utf-8"))
    confusions = json.load(open(os.path.join(MODEL_DIR, "confusions.json"), encoding="utf-8"))
    return {"pre": pre, "models": models, "metrics": metrics, "roc_data": roc_data, "confusions": confusions}


@st.cache_data
def dataset_df():
    return pd.read_csv(DATA_CSV)


def score(df_raw, pre, model, threshold=0.5):
    clean = features.clean_batch(df_raw)
    X = features.enrich(clean)
    Xp = pre.transform(X)
    probs = model.predict_proba(Xp)[:, 1]
    preds = (probs >= threshold).astype(int)
    return probs, preds


def risk_level(p):
    if p < 0.3:
        return "Low Risk", "#22c55e"
    if p < 0.6:
        return "Medium Risk", "#f59e0b"
    return "High Risk", "#ef4444"


def risk_badge(p):
    label, color = risk_level(p)
    return f'<span style="background:{color};color:white;padding:2px 12px;border-radius:12px;font-weight:600">{label}</span>'


def prediction_badge(p, threshold=0.5):
    if p >= threshold:
        return '<span style="background:#dc2626;color:white;padding:2px 12px;border-radius:12px;font-weight:700">FRAUD</span>'
    return '<span style="background:#16a34a;color:white;padding:2px 12px;border-radius:12px;font-weight:700">LEGITIMATE</span>'


def gauge(p, threshold=0.5):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%"},
            title={"text": "Fraud Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if p >= threshold else "#22c55e"},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.9, "value": threshold * 100},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def explain_signals(values):
    signals = []
    ratio = values["transaction_amount"] / max(values["avg_amount_30d"], 1e-9)
    if ratio >= 2.5:
        signals.append(f"Amount is {ratio:.1f}x the 30-day average")
    if values["tx_last_1h"] >= 4:
        signals.append(f"High burst: {int(values['tx_last_1h'])} tx in the last hour")
    if values["tx_last_24h"] >= 10:
        signals.append(f"High velocity: {int(values['tx_last_24h'])} tx in 24h")
    if values["previous_chargebacks"] >= 2:
        signals.append(f"{int(values['previous_chargebacks'])} previous chargebacks")
    if values["is_night_transaction"]:
        signals.append("Night-time transaction (22:00-06:00)")
    if values["is_international"]:
        signals.append("International transaction")
    if values["is_high_risk_merchant"]:
        signals.append("High-risk merchant category")
    if values["account_age_days"] < 90:
        signals.append("Account opened less than 90 days ago")
    return signals


def model_select_section(models):
    description = {
        "XGBoost": "Strong standalone fraud-detection model with the highest F1-score among the individual models.",
        "Random Forest": "Recall-first policy: tuned to catch as many frauds as possible, accepting more alerts for review.",
        "Stacking Ensemble": "Meta-learner over XGBoost, Random Forest and Logistic Regression. Recommended for deployment.",
    }
    choice = st.selectbox("1. Model Selection", list(models.keys()))
    st.caption(description[choice])
    return choice


def prediction_form():
    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            transaction_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
            transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=150.0, step=10.0)
            avg_amount_30d = st.number_input("Average Transaction Amount (30d)", min_value=0.01, value=300.0, step=10.0)
            account_age_days = st.number_input("Account Age (days)", min_value=1.0, value=365.0, step=1.0)
        with col2:
            previous_chargebacks = st.number_input("Previous Chargebacks", min_value=0.0, value=0.0, step=1.0)
            tx_last_1h = st.number_input("Transactions in Last 1 Hour", min_value=0.0, value=1.0, step=1.0)
            tx_last_24h = st.number_input("Transactions in Last 24 Hours", min_value=0.0, value=3.0, step=1.0)
            merchant_category = st.selectbox("Merchant Category", features.CAT_MERCHANT)
        with col3:
            country = st.selectbox("Transaction Country", features.CAT_COUNTRY)
            device_type = st.selectbox("Device Type", features.CAT_DEVICE)
            is_international = st.selectbox("International Transaction?", ["No", "Yes"])
            is_high_risk_merchant = st.selectbox("High-Risk Merchant?", ["No", "Yes"])
        submitted = st.form_submit_button("Predict Transaction", type="primary")

    values = {
        "transaction_amount": float(transaction_amount),
        "avg_amount_30d": float(avg_amount_30d),
        "account_age_days": float(account_age_days),
        "previous_chargebacks": float(previous_chargebacks),
        "tx_last_1h": float(tx_last_1h),
        "tx_last_24h": float(tx_last_24h),
        "transaction_hour": int(transaction_hour),
        "merchant_category": merchant_category,
        "country": country,
        "device_type": device_type,
        "is_international": 1 if is_international == "Yes" else 0,
        "is_high_risk_merchant": 1 if is_high_risk_merchant == "Yes" else 0,
    }
    return submitted, values


def render_result(p, model_name, values, threshold=0.5):
    label_span = prediction_badge(p, threshold)
    level_span = risk_badge(p)
    st.markdown(f"### Prediction Result")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"**Outcome:** {label_span}")
        st.markdown(f"**Risk Level:** {level_span}")
        st.markdown(f"**Model Used:** {model_name}")
        st.caption(f"Decision threshold: {threshold:.3f}")
        st.metric("Fraud Probability", f"{p * 100:.1f}%")
        st.metric("Legitimate Probability", f"{(1 - p) * 100:.1f}%")
    with c2:
        st.plotly_chart(gauge(p, threshold), use_container_width=True)

    signals = explain_signals({**values, "is_night_transaction": int((values["transaction_hour"] >= 22) or (values["transaction_hour"] <= 5))})
    with st.expander("Risk signals used by the model"):
        if signals:
            for s in signals:
                st.markdown(f"- {s}")
        else:
            st.markdown("- No elevated risk signals detected.")


def performance_page(artifacts):
    st.subheader("Model Performance")
    metrics = artifacts["metrics"]["models"]
    df_m = pd.DataFrame(metrics).T
    pct_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"] if c in df_m.columns]
    st.dataframe(df_m.style.format({c: "{:.1%}" for c in pct_cols}), use_container_width=True)

    rec = artifacts["metrics"]["recommendation"]
    st.info(f"**Recommended Model: Stacking Ensemble** - {rec}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df_m.reset_index().melt(id_vars="index", var_name="metric", value_name="score"),
            x="index", y="score", color="metric", barmode="group",
            labels={"index": "Model"},
        )
        fig.update_layout(title="Scores by model", height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        for name, data in artifacts["roc_data"].items():
            fig.add_trace(go.Scatter(x=data["fpr"], y=data["tpr"], mode="lines", name=f"{name} (AUC {data['auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="ROC curves (hold-out test set)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=360)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Confusion matrices**")
        cols = st.columns(3)
        for i, (name, cm) in enumerate(artifacts["confusions"].items()):
            with cols[i]:
                st.markdown(f"**{name}**")
                fig = go.Figure(data=go.Heatmap(z=cm, x=["Legit", "Fraud"], y=["Legit", "Fraud"], colorscale="Reds", showscale=False))
                fig.update_layout(height=220, xaxis_title="Predicted", yaxis_title="Actual")
                st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.markdown("**What drives fraud detection? (Top 10 features)**")
        tops = artifacts["metrics"]["top_features"]
        fig = px.bar(
            pd.DataFrame(tops),
            x="importance", y="feature", orientation="h",
            labels={"importance": "Importance", "feature": ""},
        )
        fig.update_layout(height=360, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Importance from the Random Forest model, aggregated over one-hot features.")


def analytics_page():
    st.subheader("Data Analytics")
    df = dataset_df()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions", f"{len(df):,}")
    k2.metric("Fraud transactions", f"{int(df['TX_FRAUD'].sum()):,}")
    k3.metric("Fraud rate", f"{df['TX_FRAUD'].mean():.2%}")
    k4.metric("Avg transaction amount", f"${df['transaction_amount'].mean():,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="transaction_amount", color=df["TX_FRAUD"].map({0: "Legitimate", 1: "Fraud"}),
                           nbins=60, log_y=True, labels={"TX_FRAUD": "Class", "transaction_amount": "Amount ($)"})
        fig.update_layout(title="Amount distribution by class", height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rate = df.groupby("transaction_hour")["TX_FRAUD"].mean().reset_index()
        fig = px.line(rate, x="transaction_hour", y="TX_FRAUD", markers=True, labels={"TX_FRAUD": "Fraud rate", "transaction_hour": "Hour"})
        fig.update_layout(title="Fraud rate by hour of day", height=340)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        rate = df.groupby("merchant_category")["TX_FRAUD"].mean().sort_values().reset_index()
        fig = px.bar(rate, x="TX_FRAUD", y="merchant_category", orientation="h", labels={"TX_FRAUD": "Fraud rate", "merchant_category": ""})
        fig.update_layout(title="Fraud rate by merchant category", height=340, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        rate = df.groupby("country")["TX_FRAUD"].mean().sort_values().reset_index()
        fig = px.bar(rate, x="country", y="TX_FRAUD", labels={"TX_FRAUD": "Fraud rate", "country": "Country"})
        fig.update_layout(title="Fraud rate by country", height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Dataset stats: {len(df):,} transactions, fraud rate {df['TX_FRAUD'].mean():.2%}. "
               f"Reference amounts are simulated daily-averaged rates; see the About tab.")


def batch_page(artifacts):
    st.subheader("Batch Detection")
    st.markdown("Upload a CSV with the required columns. A sample file is provided below.")
    st.download_button(
        "Download sample CSV",
        data=open(SAMPLE_CSV, "rb").read() if os.path.exists(SAMPLE_CSV) else b"",
        file_name="sample_transactions.csv",
        mime="text/csv",
    )

    pre = artifacts["pre"]
    models = artifacts["models"]
    model_choice = st.selectbox("Model", list(models.keys()), index=2)
    file = st.file_uploader("Upload transaction CSV", type=["csv"])

    if file is None:
        st.info("Upload a CSV to begin.")
        return

    df_in = pd.read_csv(file)
    missing = [c for c in features.RAW_FIELDS if c not in df_in.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    probs, preds = score(df_in, pre, models[model_choice], threshold=artifacts["metrics"]["models"][model_choice]["threshold"])
    out = df_in.copy()
    out["FRAUD_PROBABILITY"] = probs.round(4)
    out["PREDICTION"] = np.where(preds == 1, "FRAUD", "LEGITIMATE")
    out["RISK_LEVEL"] = [risk_level(p)[0] for p in probs]

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions scored", f"{len(out):,}")
    c2.metric("Predicted fraud", f"{int((preds == 1).sum()):,}")
    c3.metric("Predicted fraud rate", f"{(preds == 1).mean():.2%}")
    scored_csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download scored results", data=scored_csv, file_name="scored_transactions.csv", mime="text/csv")

    st.dataframe(out.sort_values("FRAUD_PROBABILITY", ascending=False).head(200), use_container_width=True)

    fig = px.histogram(out, x="FRAUD_PROBABILITY", nbins=40, labels={"FRAUD_PROBABILITY": "Fraud probability"})
    fig.update_layout(title="Fraud probability distribution", height=320)
    st.plotly_chart(fig, use_container_width=True)


def about_page(artifacts):
    st.subheader("About this project")
    m = artifacts["metrics"]
    st.markdown(
        f"""
**Business objective:** Identify potentially fraudulent transactions accurately while balancing
fraud-detection performance and false-positive errors.

- Target: `Fraud` (Legitimate / Fraud)
- Deployment models: 3 (XGBoost, Random Forest, Stacking Ensemble)
- Imbalance handling: `class_weight='balanced'`, `scale_pos_weight`, and recall-first
  threshold for Random Forest
- Validation: stratified train/test split + 3-fold cross-validated hyperparameter tuning
- Decisions: per-model threshold selected to maximise F1 (recall-first for Random Forest)
- Hold-out test set: {m['n_test']:,} transactions (train: {m['n_train']:,})

**Input features**

| Block | Features |
|---|---|
| Transaction | Transaction Hour, Transaction Amount, Average Amount (30d) |
| Account & history | Account Age (days), Previous Chargebacks |
| Behavior | Transactions in Last 1 Hour / 24 Hours |
| Merchant & context | Merchant Category, Country, Device Type, International?, High-Risk Merchant? |

Derived in the pipeline: `amount_to_avg_ratio` and `is_night_transaction`.

**Technical details**

- Preprocessing: fitted `preprocessor.joblib` (StandardScaler + OneHotEncoder) is loaded and only
  `.transform()` is called - never `.fit()` on the app side.
- Inference only: this app uses models trained and saved by `train_model.py`. No retraining,
  refitting, or retuning happens inside the app.
- The dataset is synthetic (simulated fraud patterns), generated by `data_gen.py` - perfect for
  demonstrations and workshops.

**Run**

```bash
pip install -r requirements.txt
python train_model.py      # optional; app auto-trains on first run
streamlit run app.py
```
"""
    )


def main():
    st.title("Credit Card Fraud Detection")
    st.caption("Detect potentially fraudulent transactions using trained machine learning models.")
    artifacts = get_artifacts()

    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Batch Detection", "Data Analytics", "About"])

    with tab1:
        col0, col1 = st.columns([1, 2])
        with col0:
            model_name = model_select_section(artifacts["models"])
        submitted, values = prediction_form()
        if submitted:
            df_raw = features.clean_batch(pd.DataFrame([values])[features.RAW_FIELDS])
            X = features.enrich(df_raw)
            Xp = artifacts["pre"].transform(X)
            p = float(artifacts["models"][model_name].predict_proba(Xp)[:, 1][0])
            threshold = artifacts["metrics"]["models"][model_name]["threshold"]
            render_result(p, model_name, values, threshold)
        else:
            st.markdown("_Fill the form and press **Predict Transaction**._")

    with tab2:
        batch_page(artifacts)
    with tab3:
        analytics_page()
        st.markdown("---")
        performance_page(artifacts)
    with tab4:
        about_page(artifacts)


if __name__ == "__main__":
    main()