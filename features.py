"""Технічні індикатори та формування ознак для ML-моделі."""
import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


FEATURES = [
    "rsi", "ema9", "ema21",
    "macd", "macd_signal", "macd_hist", "atr",
    "body", "range", "upper_wick", "lower_wick",
    "ret_1", "ret_2", "ret_3", "ret_5", "vol_change",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    df["rsi"] = rsi(df["close"], 14)
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["atr"] = atr(df, 14)

    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]

    for lag in [1, 2, 3, 5]:
        df[f"ret_{lag}"] = df["close"].pct_change(lag)
    df["vol_change"] = df["volume"].pct_change()

    df["target"] = (df["close"].shift(-1) > df["close"]).astype("Int64")
    return df
