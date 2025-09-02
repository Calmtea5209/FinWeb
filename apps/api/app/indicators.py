import numpy as np
import pandas as pd

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def compute_indicators(df: pd.DataFrame, specs):
    out = {}
    for spec in specs:
        name, _, param = spec.partition(":")
        p = int(param) if param else None

        if name == "sma" and p:
            out[f"sma_{p}"] = sma(df["close"], p)
        elif name == "ema" and p:
            out[f"ema_{p}"] = ema(df["close"], p)
        elif name == "rsi":
            out[f"rsi_{p or 14}"] = rsi(df["close"], p or 14)
    return pd.DataFrame(out, index=df.index)
