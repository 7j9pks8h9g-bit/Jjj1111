"""Навчання, збереження та використання ML-моделі."""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features import FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def train_model(df: pd.DataFrame) -> dict:
    data = df.dropna(subset=FEATURES + ["target"]).copy()
    if len(data) < 100:
        raise ValueError(f"Недостатньо даних для навчання: {len(data)} рядків")

    X = data[FEATURES].values.astype(float)
    y = data["target"].astype(int).values
    if len(np.unique(y)) < 2:
        raise ValueError("У даних лише один клас — модель неможливо навчити")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    accuracy = float(clf.score(X_test, y_test))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    return {
        "status": "ok",
        "accuracy": round(accuracy, 4),
        "samples": int(len(data)),
    }


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def predict(model, features_row) -> dict:
    proba = model.predict_proba([features_row])[0]
    classes = list(model.classes_)
    mapping = dict(zip(classes, proba))
    prob_up = float(mapping.get(1, 0.0))
    prob_down = float(mapping.get(0, 0.0))
    return {"up": round(prob_up, 4), "down": round(prob_down, 4)}
