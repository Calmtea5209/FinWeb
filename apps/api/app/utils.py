import numpy as np
import pandas as pd

def to_returns(close: pd.Series) -> np.ndarray:
    r = np.log(close).diff().dropna().values.astype(float)
    return r

def z_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mu = x.mean()
    sigma = x.std()
    return (x - mu) / (sigma + 1e-12)

def sliding_zdist(query: np.ndarray, series: np.ndarray, m: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view as swv
    if len(series) < m:
        return np.array([])

    windows = swv(series, window_shape=m)
    wmu = windows.mean(axis=1, keepdims=True)
    wstd = windows.std(axis=1, keepdims=True) + 1e-12
    zw = (windows - wmu) / wstd

    zq = z_norm(query[-m:])
    diffs = zw - zq
    d = np.linalg.norm(diffs, axis=1)
    return d
