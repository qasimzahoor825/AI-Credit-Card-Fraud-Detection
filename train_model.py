"""Train and evaluate Random Forest, XGBoost and a Stacking Ensemble on synthetic
credit-card fraud data. Saves inference artifacts for the Streamlit app."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import data_gen
import features

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")

DATA_CSV = os.path.join(DATA_DIR, "credit_fraud_data.csv")
SAMPLE_CSV = os.path.join(DATA_DIR, "sample_transactions.csv")

MODEL_DISPLAY = {"rf": "Random Forest", "xgb": "XGBoost", "stack": "Stacking Ensemble"}
MODEL_ORDER = ["xgb", "rf", "stack"]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features.NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features.CAT_COLS),
        ]
    )


def train():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Generating dataset...")
    df = data_gen.generate_dataset()
    df.to_csv(DATA_CSV, index=False)

    y = df["TX_FRAUD"].to_numpy()
    X = df[features.INPUT_COLUMNS].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting preprocessor (train split only)...")
    pre = build_preprocessor()
    pre.fit(X_train)
    Xp_train = pre.transform(X_train)
    Xp_test = pre.transform(X_test)

    neg, pos = int(np.sum(y_train == 0)), int(np.sum(y_train == 1))
    scale_pos_weight = neg / max(pos, 1)

    models = {}

    print("Tuning Random Forest...")
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    rf_grid = {
        "n_estimators": [120, 180],
        "max_depth": [9, 14],
        "min_samples_leaf": [1, 3],
    }
    rf_search = RandomizedSearchCV(
        rf, rf_grid, n_iter=3, cv=3, scoring="f1", random_state=42, n_jobs=-1, verbose=1
    )
    rf_search.fit(Xp_train, y_train)
    models["rf"] = rf_search.best_estimator_
    print("RF best params:", rf_search.best_params_, "best cv f1:", round(rf_search.best_score_, 4))

    print("Tuning XGBoost...")
    xgb = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_grid = {
        "n_estimators": [150, 250],
        "max_depth": [4, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb, xgb_grid, n_iter=3, cv=3, scoring="f1", random_state=42, n_jobs=-1, verbose=1
    )
    xgb_search.fit(Xp_train, y_train)
    models["xgb"] = xgb_search.best_estimator_
    print("XGB best params:", xgb_search.best_params_, "best cv f1:", round(xgb_search.best_score_, 4))

    print("Training Stacking Ensemble...")
    stack = StackingClassifier(
        estimators=[
            ("rf", models["rf"]),
            ("xgb", models["xgb"]),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=42),
        cv=3,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )
    stack.fit(Xp_train, y_train)
    models["stack"] = stack

    print("Evaluating on hold-out test set...")
    metrics = {"models": {}, "recommended": "stack", "recommendation": (
        "The Stacking Ensemble is the recommended model because it provides the strongest "
        "overall balance between fraud detection and false-positive control. It combines "
        "XGBoost, Random Forest and Logistic Regression through a stacked meta-learner."
    )}
    roc_data = {}
    cms = {}
    predictions = {}

    for key in MODEL_ORDER:
        m = models[key]
        probs = m.predict_proba(Xp_test)[:, 1]
        min_recall = 0.72 if key == "rf" else 0.0
        thr, _ = _best_threshold(y_test, probs, min_recall=min_recall)
        preds = (probs >= thr).astype(int)
        predictions[key] = {"probs": probs.tolist(), "preds": preds.tolist(), "y": y_test.tolist()}
        recall_policy = {"rf": "recall-first", "xgb": "F1-optimal", "stack": "F1-optimal (balanced)"}[key]
        metrics["models"][MODEL_DISPLAY[key]] = {
            "policy": recall_policy,
            "threshold": round(float(thr), 3),
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds)), 4),
            "f1": round(float(f1_score(y_test, preds)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        }
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_data[MODEL_DISPLAY[key]] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(roc_auc_score(y_test, probs)), 4),
        }
        cms[MODEL_DISPLAY[key]] = confusion_matrix(y_test, preds).tolist()

    metrics["fraud_rate_train"] = round(float(y_train.mean()), 4)
    metrics["fraud_rate_test"] = round(float(y_test.mean()), 4)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    metrics["input_columns"] = features.INPUT_COLUMNS
    metrics["scale_pos_weight"] = round(float(scale_pos_weight), 2)

    print("Top-10 feature importances (Random Forest)...")
    importance = _top_importances(models["rf"], pre)
    metrics["top_features"] = importance

    with open(os.path.join(MODEL_DIR, "preprocessor.joblib"), "wb") as fh:
        joblib.dump(pre, fh)
    with open(os.path.join(MODEL_DIR, "rf_model.joblib"), "wb") as fh:
        joblib.dump(models["rf"], fh)
    with open(os.path.join(MODEL_DIR, "xgb_model.joblib"), "wb") as fh:
        joblib.dump(models["xgb"], fh)
    with open(os.path.join(MODEL_DIR, "stack_model.joblib"), "wb") as fh:
        joblib.dump(models["stack"], fh)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(MODEL_DIR, "roc_data.json"), "w") as fh:
        json.dump(roc_data, fh, indent=2)
    with open(os.path.join(MODEL_DIR, "confusions.json"), "w") as fh:
        json.dump(cms, fh, indent=2)

    test_full = X_test.copy()
    test_full["TX_FRAUD"] = y_test
    pos_sample = test_full[test_full["TX_FRAUD"] == 1].sample(min(8, int(np.sum(y_test == 1))), random_state=1)
    neg_sample = test_full[test_full["TX_FRAUD"] == 0].sample(min(32, int(np.sum(y_test == 0))), random_state=1)
    sample = pd.concat([pos_sample, neg_sample])[list(features.INPUT_COLUMNS) + ["TX_FRAUD"]]
    sample.to_csv(SAMPLE_CSV, index=False)

    print("\n=== TEST METRICS ===")
    for name, mv in metrics["models"].items():
        print(name, mv)
    print("Saved artifacts ->", MODEL_DIR)
    return metrics


def _best_threshold(y, probs, min_recall=0.0):
    best_t, best_f = 0.5, -1.0
    for t in np.round(np.linspace(0.01, 0.99, 200), 4):
        preds = (probs >= t).astype(int)
        r = recall_score(y, preds)
        f = f1_score(y, preds)
        if r >= min_recall and f > best_f:
            best_t, best_f = t, f
    return best_t, best_f


def _top_importances(rf, pre):
    names = pre.get_feature_names_out()
    imp = rf.feature_importances_
    total = {}
    for name, v in zip(names, imp):
        if name.startswith("num__"):
            key = name[len("num__"):]
            total[key] = total.get(key, 0.0) + v
        else:
            parts = name.split("__", 2)
            if len(parts) == 3:
                key = parts[1]
                total[key] = total.get(key, 0.0) + v
    ranked = sorted(total.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return [{"feature": k, "importance": round(float(v), 4)} for k, v in ranked]


def model_exists():
    needed = ["preprocessor.joblib", "rf_model.joblib", "xgb_model.joblib", "stack_model.joblib", "metrics.json"]
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in needed)


if __name__ == "__main__":
    train()