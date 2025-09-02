import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime

from .db import Base, engine, get_db, get_db_safe
from .models import User, Symbol, OHLCV, BacktestRun
from .schemas import IndicatorRequest, SimilarSearchRequest, BacktestSubmitRequest
from .indicators import compute_indicators
from .utils import to_returns, z_norm, sliding_zdist
from .tasks import enqueue_backtest
import httpx
import math
from datetime import datetime, timezone
from typing import Tuple

Base.metadata.create_all(bind=engine)

app = FastAPI(title="FinLab API", version="0.1.1")

# Derive allowed web origin from configured dev port to avoid CORS mismatches
WEB_PORT = os.getenv("WEB_PORT", os.getenv("PORT", "3000"))
origins = [
    f"http://localhost:{WEB_PORT}",
    f"http://127.0.0.1:{WEB_PORT}",
]
# Optionally allow an explicit origin override
EXTRA_ORIGIN = os.getenv("NEXT_PUBLIC_WEB_ORIGIN")
if EXTRA_ORIGIN:
    origins.append(EXTRA_ORIGIN)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/chart/ohlcv")
def get_ohlcv(symbol: str, limit: int = 500, db: Session = Depends(get_db)):
    sym = db.query(Symbol).filter(Symbol.code == symbol).first()
    if not sym:
        raise HTTPException(404, "symbol not found")
    q = f"""SELECT ts, open, high, low, close, volume
              FROM ohlcv WHERE symbol_id={sym.id}
              ORDER BY ts DESC LIMIT {int(limit)}"""
    df = pd.read_sql(q, db.bind, parse_dates=["ts"]).sort_values("ts")
    return {
        "symbol": symbol,
        "items": [
            {
                "ts": r["ts"].isoformat(),
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": float(r["volume"]),
            } for _, r in df.iterrows()
        ]
    }

def _normalize_tf(tf: str) -> str:
    tf = (tf or "").lower()
    mapping = {
        "5m": "5 minutes",
        "15m": "15 minutes",
        "1h": "1 hour",
        "1d": "1 day",
        "d": "1 day",
        "day": "1 day",
    }
    return mapping.get(tf, "1 day")

def _tf_seconds(tf_text: str) -> int:
    if tf_text == "5 minutes":
        return 5 * 60
    if tf_text == "15 minutes":
        return 15 * 60
    if tf_text == "1 hour":
        return 60 * 60
    return 24 * 60 * 60

@app.get("/chart/ohlcv_agg")
def get_ohlcv_agg(symbol: str, tf: str = "1d", limit: int = 500, db: Session = Depends(get_db)):
    """Aggregate OHLCV to a given timeframe using time_bucket.

    Note: For intraday tfs (e.g., 5m, 15m, 1h), you must have underlying
    intraday rows in the ohlcv table; otherwise each day becomes a single bucket.
    """
    sym = db.query(Symbol).filter(Symbol.code == symbol).first()
    if not sym:
        raise HTTPException(404, "symbol not found")

    tf_text = _normalize_tf(tf)
    tf_sec = _tf_seconds(tf_text)

    # Validate requested timeframe is not finer than base data resolution
    from sqlalchemy import text as sql_text
    step_q = sql_text(
        """
        SELECT MIN(EXTRACT(EPOCH FROM ts - prev_ts)) AS min_step
        FROM (
            SELECT ts, LAG(ts) OVER (ORDER BY ts) AS prev_ts
            FROM ohlcv
            WHERE symbol_id = :sid
        ) s
        WHERE prev_ts IS NOT NULL
        """
    )
    step_row = db.execute(step_q, {"sid": sym.id}).mappings().first()
    min_step = step_row["min_step"] if step_row else None
    if min_step is not None and tf_sec < int(min_step):
        raise HTTPException(
            400,
            detail=f"timeframe '{tf}' too fine for data resolution (~{int(min_step)}s)"
        )

    # Build a safe SQL with fixed tf_text from allowlist
    q = sql_text(f"""
        WITH b AS (
            SELECT time_bucket(INTERVAL '{tf_text}', ts) AS bucket, ts, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol_id = :sid
        ), a AS (
            SELECT bucket,
                   FIRST_VALUE(open) OVER (PARTITION BY bucket ORDER BY ts ASC
                       ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS o,
                   MAX(high)    OVER (PARTITION BY bucket) AS h,
                   MIN(low)     OVER (PARTITION BY bucket) AS l,
                   LAST_VALUE(close) OVER (PARTITION BY bucket ORDER BY ts ASC
                       ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS c,
                   SUM(volume)  OVER (PARTITION BY bucket) AS v
            FROM b
        )
        SELECT DISTINCT bucket, o AS open, h AS high, l AS low, c AS close, v AS volume
        FROM a
        ORDER BY bucket DESC
        LIMIT :lim
    """)
    df = pd.read_sql(q, db.bind, params={"sid": sym.id, "lim": int(limit)}, parse_dates=["bucket"]).sort_values("bucket")
    return {
        "symbol": symbol,
        "tf": tf_text,
        "items": [
            {
                "ts": r["bucket"].isoformat(),
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": float(r.get("volume") or 0.0),
            } for _, r in df.iterrows()
        ]
    }

# Live (no-DB) OHLCV via Yahoo Finance v8 chart API
def _tf_to_yahoo(tf: str) -> tuple[str, str]:
    tf = (tf or "1d").lower()
    if tf == "5m":
        return ("5m", "60d")    # up to ~60 days of 5-min bars (Yahoo limit)
    if tf == "15m":
        return ("15m", "60d")   # ~60 days of 15-min bars (Yahoo limit)
    if tf == "1h":
        return ("60m", "1y")    # ~1 year of 60-min bars
    return ("1d", "5y")         # daily: 5 years

def _fallback_ranges_for_interval(interval: str) -> list[str]:
    if interval == "5m":
        return ["60d", "30d", "5d", "1d"]
    if interval == "15m":
        return ["60d", "30d", "5d", "1d"]
    if interval == "60m":
        return ["1y", "6mo", "3mo", "1mo"]
    # daily and others
    return ["5y", "2y", "1y", "6mo", "3mo", "1mo"]

@app.get("/chart/ohlcv_live")
def get_ohlcv_live(symbol: str, tf: str = "1d", rng: str | None = None, limit: int = 500):
    interval, default_range = _tf_to_yahoo(tf)
    requested_range = rng or default_range
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FinLabBot/1.0)",
        "Accept": "application/json",
    }

    tried = []
    last_err: Exception | None = None
    used_range = None
    for rname in [requested_range] + [x for x in _fallback_ranges_for_interval(interval) if x != requested_range]:
        params = {
            "interval": interval,
            "range": rname,
            "includePrePost": "false",
            "events": "div,splits",
        }
        tried.append(rname)
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0, read=20.0)) as client:
                resp = client.get(url, params=params, headers=headers)
                if resp.status_code in (400, 404, 422, 429) and rname != tried[-1]:
                    # Try next fallback range
                    last_err = httpx.HTTPStatusError("invalid range", request=resp.request, response=resp)
                    continue
                resp.raise_for_status()
                j = resp.json()
                used_range = rname
                break
        except httpx.HTTPError as e:
            last_err = e
            # if network error, break immediately
            if not isinstance(e, httpx.HTTPStatusError):
                break
            # for other 4xx/5xx, attempt next fallback if any
            continue
    else:
        # exhausted all fallbacks
        raise HTTPException(502, f"upstream fetch failed: {last_err}")

    result = ((j or {}).get("chart") or {}).get("result") or []
    if not result:
        err = ((j or {}).get("chart") or {}).get("error")
        raise HTTPException(502, f"upstream returned no data: {err}")

    res = result[0]
    ts = (res.get("timestamp") or [])
    ind = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    open_a = ind.get("open") or []
    high_a = ind.get("high") or []
    low_a  = ind.get("low") or []
    close_a= ind.get("close") or []
    vol_a  = ind.get("volume") or []

    items = []
    for i in range(min(len(ts), len(open_a), len(high_a), len(low_a), len(close_a))):
        o = open_a[i]
        h = high_a[i]
        l = low_a[i]
        c = close_a[i]
        if o is None or h is None or l is None or c is None:
            continue
        o = float(o); h = float(h); l = float(l); c = float(c)
        # Skip NaN/inf
        if not all(math.isfinite(x) for x in (o, h, l, c)):
            continue
        # Ensure high/low envelope contains open/close (some feeds have rounding issues)
        h = max(h, o, c)
        l = min(l, o, c)
        items.append({
            # Use UTC timestamps to avoid timezone-related gaps in UI
            "ts": datetime.fromtimestamp(ts[i], tz=timezone.utc).isoformat(),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": float(vol_a[i] or 0.0),
        })

    # keep only the last N points if limit provided
    if limit and len(items) > int(limit):
        items = items[-int(limit):]

    return {"symbol": symbol, "tf": tf, "range": used_range or requested_range, "items": items}

@app.get("/chart/meta")
def get_chart_meta(symbol: str, db: Session = Depends(get_db)):
    sym = db.query(Symbol).filter(Symbol.code == symbol).first()
    if not sym:
        raise HTTPException(404, "symbol not found")
    from sqlalchemy import text as sql_text
    step_q = sql_text(
        """
        SELECT MIN(EXTRACT(EPOCH FROM ts - prev_ts)) AS min_step
        FROM (
            SELECT ts, LAG(ts) OVER (ORDER BY ts) AS prev_ts
            FROM ohlcv
            WHERE symbol_id = :sid
        ) s
        WHERE prev_ts IS NOT NULL
        """
    )
    row = db.execute(step_q, {"sid": sym.id}).mappings().first()
    min_step = int(row["min_step"]) if row and row["min_step"] is not None else None
    return {"symbol": symbol, "min_step_seconds": min_step}

@app.post("/indicators/run")
def indicators(req: IndicatorRequest, db: Session = Depends(get_db)):
    sym = db.query(Symbol).filter(Symbol.code == req.symbol).first()
    if not sym:
        raise HTTPException(404, "symbol not found")
    q = f"""SELECT ts, close FROM ohlcv WHERE symbol_id={sym.id}
              ORDER BY ts DESC LIMIT {int(req.limit)}"""
    df = pd.read_sql(q, db.bind, parse_dates=["ts"]).sort_values("ts").set_index("ts")
    out = compute_indicators(df, req.indicators).fillna(method="bfill").fillna(method="ffill")
    return {
        "symbol": req.symbol,
        "indicators": out.reset_index().rename(columns={"index":"ts"}).to_dict(orient="records")
    }

@app.post("/similar/search")
def similar(req: SimilarSearchRequest, db: Session | None = Depends(get_db_safe)):
    # Try to resolve symbol in DB if available
    sym = None
    if db is not None:
        try:
            sym = db.query(Symbol).filter(Symbol.code == req.symbol).first()
        except Exception:
            sym = None

    def yahoo_df(symbol: str, interval: str = "1d", y_range: str = "5y") -> pd.DataFrame:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FinLabBot/1.0)", "Accept": "application/json"}
        params = {"interval": interval, "range": y_range, "includePrePost": "false", "events": "div,splits"}
        with httpx.Client(timeout=httpx.Timeout(10.0, read=20.0)) as client:
            r = client.get(url, params=params, headers=headers)
            r.raise_for_status()
            j = r.json()
        result = ((j or {}).get("chart") or {}).get("result") or []
        if not result:
            raise HTTPException(502, "upstream returned no data")
        res = result[0]
        ts = res.get("timestamp") or []
        ind = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        close_a = ind.get("close") or []
        rows = []
        for i in range(min(len(ts), len(close_a))):
            c = close_a[i]
            if c is None or not math.isfinite(float(c)):
                continue
            rows.append({"ts": datetime.fromtimestamp(ts[i], tz=timezone.utc), "close": float(c)})
        if not rows:
            return pd.DataFrame(columns=["ts","close"]).set_index("ts")
        df = pd.DataFrame(rows).sort_values("ts").set_index("ts")
        return df

    # Try DB first if symbol exists in DB
    qdf = pd.DataFrame()
    if sym is not None:
        qdf = pd.read_sql(
            f"""SELECT ts, close FROM ohlcv WHERE symbol_id={sym.id}
                 AND ts >= %(s)s AND ts <= %(e)s ORDER BY ts""",
            db.bind, params={"s": req.start, "e": req.end}, parse_dates=["ts"]
        ).set_index("ts")
        # Normalize to UTC to align with Yahoo-based data
        if not getattr(qdf.index, 'tz', None):
            try:
                qdf.index = qdf.index.tz_localize('UTC')
            except Exception:
                pass

    # Fallback to Yahoo if DB has no/insufficient data
    if qdf.empty or len(qdf) < req.m + 2:
        try:
            full = yahoo_df(req.symbol, interval="1d", y_range="10y")
        except httpx.HTTPError as e:
            raise HTTPException(502, f"upstream fetch failed: {e}")
        # Ensure both sides are UTC-aware for comparison
        start_utc = pd.to_datetime(req.start, utc=True)
        end_utc = pd.to_datetime(req.end, utc=True)
        qdf = full[(full.index >= start_utc) & (full.index <= end_utc)]
        if qdf.empty or len(qdf) < req.m + 2:
            raise HTTPException(400, "query segment too short / no data")

    # Build query vector (last m returns inside selected window)
    q_ret = to_returns(qdf["close"])[-req.m:]
    zq = z_norm(q_ret)
    # Determine query start/end timestamps for overlap filtering
    try:
        q_end_ts = qdf.index[-1]
        q_start_ts = qdf.index[-req.m]
    except Exception:
        q_start_ts = qdf.index[0]
        q_end_ts = qdf.index[-1]

    results = []
    # Build candidate universe
    req_codes = []
    if getattr(req, 'universe', None):
        try:
            req_codes = [str(c).strip() for c in req.universe if str(c).strip()]
        except Exception:
            req_codes = []
    db_symbols = []
    if not req_codes and db is not None:
        try:
            db_symbols = db.query(Symbol).all()
        except Exception:
            db_symbols = []
    codes = req_codes or ([s.code for s in db_symbols] if db_symbols else [req.symbol])
    # Deduplicate and cap size to avoid rate limiting
    seen = set()
    codes = [c for c in codes if not (c in seen or seen.add(c))][:50]
    for code in codes:
        if db_symbols:
            s = next((x for x in db_symbols if x.code == code), None)
            if s is not None:
                sdf = pd.read_sql(
                    f"""SELECT ts, close FROM ohlcv WHERE symbol_id={s.id} ORDER BY ts""",
                    db.bind, parse_dates=["ts"]
                ).set_index("ts")
                if not getattr(sdf.index, 'tz', None):
                    try:
                        sdf.index = sdf.index.tz_localize('UTC')
                    except Exception:
                        pass
            else:
                sdf = pd.DataFrame()
        else:
            # Yahoo fallback for target universe
            try:
                sdf = yahoo_df(code, interval="1d", y_range="10y")
            except httpx.HTTPError:
                continue
        if len(sdf) < req.m + 10:
            continue
        s_ret = to_returns(sdf["close"])
        D = sliding_zdist(zq, s_ret, req.m)
        if D.size == 0:
            continue
        k = min(req.top, len(D))
        idx = np.argpartition(D, k-1)[:k]
        idx = idx[np.argsort(D[idx])]
        for i in idx:
            end_pos = i + req.m - 1
            start_ts = sdf.index[i+1]
            end_ts = sdf.index[i+req.m]
            closes = sdf["close"].values
            # Skip trivial self-matches that overlap the query window when searching within the same symbol
            if code == req.symbol:
                if not (end_ts <= q_start_ts or start_ts >= q_end_ts):
                    continue
            def fwd(h):
                if end_pos + 1 + h < len(closes):
                    return float(np.log(closes[end_pos+1+h]) - np.log(closes[end_pos+1]))
                return None
            results.append({
                "symbol": code,
                "start_time": start_ts.isoformat(),
                "end_time": end_ts.isoformat(),
                "distance": float(D[i]),
                "forward": {"1": fwd(1), "5": fwd(5), "20": fwd(20)}
            })
    results.sort(key=lambda x: x["distance"])
    return {"items": results[:req.top]}

# --- Pattern classification (triangles) ---
def _yahoo_ohlc_df(symbol: str, interval: str = "1d", y_range: str = "5y") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinLabBot/1.0)", "Accept": "application/json"}
    params = {"interval": interval, "range": y_range, "includePrePost": "false", "events": "div,splits"}
    with httpx.Client(timeout=httpx.Timeout(10.0, read=20.0)) as client:
        r = client.get(url, params=params, headers=headers)
        r.raise_for_status()
        j = r.json()
    result = ((j or {}).get("chart") or {}).get("result") or []
    if not result:
        raise HTTPException(502, "upstream returned no data")
    res = result[0]
    ts = res.get("timestamp") or []
    ind = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    open_a = ind.get("open") or []
    high_a = ind.get("high") or []
    low_a  = ind.get("low") or []
    close_a= ind.get("close") or []
    rows = []
    n = min(len(ts), len(open_a), len(high_a), len(low_a), len(close_a))
    for i in range(n):
        o, h, l, c = open_a[i], high_a[i], low_a[i], close_a[i]
        if any(v is None for v in (o,h,l,c)):
            continue
        o = float(o); h = float(h); l = float(l); c = float(c)
        if not all(math.isfinite(x) for x in (o,h,l,c)):
            continue
        rows.append({
            "ts": datetime.fromtimestamp(ts[i], tz=timezone.utc),
            "open": o, "high": h, "low": l, "close": c,
        })
    if not rows:
        return pd.DataFrame(columns=["ts","open","high","low","close"]).set_index("ts")
    df = pd.DataFrame(rows).sort_values("ts").set_index("ts")
    return df

def _linreg(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, intercept, r2."""
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    s, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = s*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-12
    r2 = 1 - ss_res/ss_tot
    return float(s), float(b), float(max(0.0, min(1.0, r2)))

def _find_peaks_troughs(df: pd.DataFrame, w: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    peaks = []
    troughs = []
    for i in range(w, n-w):
        if h[i] == max(h[i-w:i+w+1]):
            peaks.append(i)
        if l[i] == min(l[i-w:i+w+1]):
            troughs.append(i)
    return np.array(peaks), np.array(troughs)

def _detect_triangle(df: pd.DataFrame) -> Tuple[str, float, dict]:
    # Require minimum bars
    if len(df) < 20:
        return "unknown", 0.0, {"reason":"too_few_bars"}
    peaks, troughs = _find_peaks_troughs(df, w=2)
    n = len(df)
    price_scale = float(np.median(df["close"].values)) or 1.0

    method = "extrema"
    if len(peaks) < 3 or len(troughs) < 3:
        # Fallback: quantile-based envelope regression to always get lines
        method = "quantile"
        h = df["high"].values
        l = df["low"].values
        qh = np.quantile(h, 0.8) if n >= 5 else max(h)
        ql = np.quantile(l, 0.2) if n >= 5 else min(l)
        p_idx = np.where(h >= qh)[0]
        t_idx = np.where(l <= ql)[0]
        if len(p_idx) < 3:
            p_idx = np.argsort(h)[-min(3, n):]
        if len(t_idx) < 3:
            t_idx = np.argsort(l)[:min(3, n)]
    else:
        # Use last K extrema for lines
        K = 6
        p_idx = peaks[-K:]
        t_idx = troughs[-K:]

    # Normalize x over [0,1] so slope represents total change over window
    xh = (p_idx.astype(float)) / max(1.0, (n - 1))
    xl = (t_idx.astype(float)) / max(1.0, (n - 1))
    yh = df["high"].values[p_idx]
    yl = df["low"].values[t_idx]
    sh, bh, r2h = _linreg(xh, yh)
    sl, bl, r2l = _linreg(xl, yl)
    nsh = sh / price_scale
    nsl = sl / price_scale
    # Range contraction
    wlen = max(10, int(len(df)*0.3))
    rng_start = float(df["high"].iloc[:wlen].max() - df["low"].iloc[:wlen].min())
    rng_end   = float(df["high"].iloc[-wlen:].max() - df["low"].iloc[-wlen:].min())
    contraction = rng_end / (rng_start + 1e-9)
    # Heuristics with soft confidence scoring
    # Thresholds tuned for normalized x in [0,1]
    s_flat = 1.5e-3   # ~0.15% considered flat
    s_min  = 3.5e-3   # ~0.35% considered meaningful slope

    def pos_sig(x: float, k: float = 1.0) -> float:
        x = max(0.0, x)
        return float(np.tanh((x / max(1e-9, s_min)) * k))

    def flat_sig(x: float, k: float = 1.0) -> float:
        return float(1.0 - np.tanh((abs(x) / max(1e-9, s_flat)) * k))

    contraction_conf = float(max(0.0, min(1.0, 1.0 - contraction)))
    fit_conf = float(max(0.0, min(1.0, (r2h + r2l) / 2.0)))

    sym_conf = float(np.mean([
        pos_sig(-nsh),
        pos_sig(nsl),
        fit_conf,
        contraction_conf,
    ]))
    asc_conf = float(np.mean([
        flat_sig(nsh),
        pos_sig(nsl),
        fit_conf,
        contraction_conf,
    ]))
    desc_conf = float(np.mean([
        flat_sig(nsl),
        pos_sig(-nsh),
        fit_conf,
        contraction_conf,
    ]))

    conf_map = {"sym_triangle": sym_conf, "asc_triangle": asc_conf, "desc_triangle": desc_conf}
    label = max(conf_map, key=conf_map.get)
    conf = conf_map[label]
    if conf < 0.25:
        label, conf = "unknown", 0.0
    meta = {
        "slope_high": nsh, "slope_low": nsl,
        "r2_high": r2h, "r2_low": r2l,
        "contraction": contraction,
        "peaks": len(peaks), "troughs": len(troughs),
        "method": method,
        "fit_conf": fit_conf,
        "contraction_conf": contraction_conf,
        "sym_conf": sym_conf,
        "asc_conf": asc_conf,
        "desc_conf": desc_conf,
    }
    return label, conf, meta

class PatternRequest(BaseModel):
    symbol: str
    start: str
    end: str
    tf: str = "1d"

@app.post("/pattern/classify")
def pattern_classify(req: PatternRequest):
    # Fetch daily OHLC between start/end (from Yahoo)
    try:
        full = _yahoo_ohlc_df(req.symbol, interval="1d", y_range="10y")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"upstream fetch failed: {e}")
    if full.empty:
        raise HTTPException(400, "no data")
    start_utc = pd.to_datetime(req.start, utc=True)
    end_utc = pd.to_datetime(req.end, utc=True)
    df = full[(full.index >= start_utc) & (full.index <= end_utc)]
    if len(df) < 20:
        raise HTTPException(400, "window too short (min 20 bars)")
    label, conf, meta = _detect_triangle(df)
    return {"symbol": req.symbol, "start": req.start, "end": req.end, "label": label, "confidence": conf, "meta": meta}

@app.post("/backtest/submit")
def backtest_submit(req: BacktestSubmitRequest, db: Session = Depends(get_db)):
    job_id = enqueue_backtest(req.config.dict())
    return {"job_id": job_id}

@app.get("/backtest/status")
def backtest_status(job_id: str, db: Session = Depends(get_db)):
    run = db.get(BacktestRun, job_id)
    if not run:
        raise HTTPException(404, "job not found")
    return {"job_id": job_id, "status": run.status}

@app.get("/backtest/report")
def backtest_report(job_id: str, db: Session = Depends(get_db)):
    run = db.get(BacktestRun, job_id)
    if not run:
        raise HTTPException(404, "job not found")
    return {"job_id": job_id, "status": run.status, "report": run.report_json}
