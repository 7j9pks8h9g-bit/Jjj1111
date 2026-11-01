"""Логіка одного циклу прогнозу: fetch → features → predict → signal."""
import logging
from datetime import datetime, timezone

import pandas as pd

from data_fetcher import fetch_klines
from features import build_features, FEATURES
from model import load_model, predict

log = logging.getLogger("bot")

STATE = {
    "last_signal": None,
    "last_update": None,
    "last_error": None,
}


def _heuristic(features: pd.Series) -> dict:
    score = 0.0
    rsi_val = float(features["rsi"])
    if rsi_val < 30:
        score += 0.3
    elif rsi_val > 70:
        score -= 0.3

    score += 0.4 if float(features["ret_1"]) > 0 else -0.4
    score += 0.2 if float(features["macd_hist"]) > 0 else -0.2
    score += 0.1 if float(features["ema9"]) > float(features["ema21"]) else -0.1

    prob_up = 0.5 + max(-0.4, min(0.4, score * 0.5))
    return {"up": round(prob_up, 4), "down": round(1 - prob_up, 4)}


def run_cycle(symbol: str = "BTCUSDT") -> dict | None:
    try:
        df = fetch_klines(symbol=symbol, interval="5m", limit=500)
        feats = build_features(df)

        valid = feats.dropna(subset=FEATURES)
        if valid.empty:
            raise ValueError("Недостатньо даних для розрахунку ознак")

        latest = valid.iloc[-1]
        feature_vec = latest[FEATURES].values.astype(float)

        model = load_model()
        if model is not None:
            probs = predict(model, feature_vec)
            source = "model"
        else:
            probs = _heuristic(latest)
            source = "heuristic"

        direction = "UP" if probs["up"] >= probs["down"] else "DOWN"

        signal = {
            "symbol": symbol,
            "interval": "5m",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "next_candle_direction": direction,
            "probability_up": probs["up"],
            "probability_down": probs["down"],
            "source": source,
            "last_close": float(latest["close"]),
            "rsi": round(float(latest["rsi"]), 2),
            "macd_hist": round(float(latest["macd_hist"]), 6),
            "ema9": round(float(latest["ema9"]), 2),
            "ema21": round(float(latest["ema21"]), 2),
        }
        STATE["last_signal"] = signal
        STATE["last_update"] = signal["time_utc"]
        STATE["last_error"] = None
        log.info("Signal: %s", signal)
        return signal

    except Exception as e:
        STATE["last_error"] = str(e)
        log.exception("Помилка в циклі прогнозу")
        return None
