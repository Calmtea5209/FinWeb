import os
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime

from .db import Base, engine, get_db, get_db_safe
from .models import User, UserAuth, Symbol, OHLCV, BacktestRun
from .schemas import IndicatorRequest, SimilarSearchRequest, BacktestSubmitRequest, AuthRegister, AuthLogin, AuthToken, UserOut
from .indicators import compute_indicators
from .utils import to_returns, z_norm, sliding_zdist
from .tasks import enqueue_backtest
from .auth import hash_password, verify_password, create_access_token, decode_token
import httpx
import math
from datetime import datetime, timezone
from typing import Tuple, List, Optional
import re
from urllib.parse import urlparse

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

# --- Auth endpoints ---
@app.post("/auth/register", response_model=UserOut)
def auth_register(req: AuthRegister, db: Session = Depends(get_db)):
    email = str(req.email).strip().lower()
    if not email:
        raise HTTPException(400, "invalid email")
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        # if user exists but no auth, allow setting password once
        ua = db.query(UserAuth).filter(UserAuth.user_id == existing.id).first()
        if ua:
            raise HTTPException(400, "user already exists")
        # set password for existing user
        try:
            ph = hash_password(req.password)
        except ValueError as e:
            raise HTTPException(400, str(e))
        ua = UserAuth(user_id=existing.id, password_hash=ph)
        db.add(ua)
        db.commit()
        db.refresh(existing)
        return UserOut(id=existing.id, email=existing.email, tz=existing.tz)
    # create new user
    try:
        ph = hash_password(req.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # Always use Asia/Taipei for user timezone per product decision
    u = User(email=email, tz="Asia/Taipei")
    db.add(u)
    db.flush()
    ua = UserAuth(user_id=u.id, password_hash=ph)
    db.add(ua)
    db.commit()
    db.refresh(u)
    return UserOut(id=u.id, email=u.email, tz=u.tz)


@app.post("/auth/login", response_model=AuthToken)
def auth_login(req: AuthLogin, db: Session = Depends(get_db)):
    email = str(req.email).strip().lower()
    u = db.query(User).filter(User.email == email).first()
    if not u:
        raise HTTPException(401, "invalid credentials")
    ua = db.query(UserAuth).filter(UserAuth.user_id == u.id).first()
    if not ua or not verify_password(req.password, ua.password_hash):
        raise HTTPException(401, "invalid credentials")
    token, exp = create_access_token(u.id, u.email)
    # update last login
    try:
        ua.last_login_at = datetime.now(timezone.utc)
        db.add(ua)
        db.commit()
    except Exception:
        db.rollback()
    return AuthToken(access_token=token, token_type="bearer", expires_at=exp.isoformat(), user=UserOut(id=u.id, email=u.email, tz=u.tz))


def _user_from_auth_header(authorization: str | None, db: Session) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "invalid token")
    uid = payload.get("sub")
    try:
        uid = int(uid)
    except Exception:
        raise HTTPException(401, "invalid token")
    u = db.query(User).filter(User.id == uid).first()
    if not u:
        raise HTTPException(401, "user not found")
    return u


@app.get("/auth/me", response_model=UserOut)
def auth_me(Authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    u = _user_from_auth_header(Authorization, db)
    return UserOut(id=u.id, email=u.email, tz=u.tz)

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
    # Restrict candidate search to recent N years (default 5)
    lookback_years = getattr(req, 'lookback_years', 5) or 5
    # Use now(tz='UTC') to avoid tz_localize edge cases
    cutoff_utc = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=int(lookback_years))

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
                try:
                    sdf = sdf[sdf.index >= cutoff_utc]
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
        if not sdf.empty:
            try:
                sdf = sdf[sdf.index >= cutoff_utc]
            except Exception:
                pass
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

# --- Sentiment summary ---
def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    s = series.astype(float)
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean() + 1e-12
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

class SentimentRequest(BaseModel):
    symbol: str
    universe: Optional[List[str]] = None

@app.post("/sentiment/summary")
def sentiment_summary(req: SentimentRequest):
    # Individual symbol sentiment (daily, 6 months)
    sym = req.symbol
    try:
        df = _yahoo_ohlc_df(sym, interval="1d", y_range="6mo")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"upstream fetch failed: {e}")
    if df.empty:
        raise HTTPException(400, "no data")
    close = df["close"].astype(float)
    ret = close.pct_change().dropna()
    rsi14 = _rsi(close, 14).iloc[-1]
    # 60-bar normalized slope on log price
    last = close.tail(min(60, len(close)))
    x = (pd.RangeIndex(len(last)).values.astype(float))
    x = (x - x.min()) / max(1.0, (x.max() - x.min()))
    y = np.log(last.values)
    s, b = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    slope_norm = float(s)
    vol = float(ret.tail(min(60, len(ret))).std())
    # score 0..1 from RSI and slope
    score = float(np.clip(0.5*(rsi14/100.0) + 0.5*(np.tanh(slope_norm*8)+1)/2, 0, 1))
    if score >= 0.7:
        sym_label = "極度偏多"
    elif score >= 0.55:
        sym_label = "偏多"
    elif score <= 0.3:
        sym_label = "極度偏空"
    elif score <= 0.45:
        sym_label = "偏空"
    else:
        sym_label = "中性"

    # Environment sentiment across a small universe
    universe = req.universe or [sym]
    # dedupe and cap to avoid rate limits
    uniq = []
    for c in universe:
        c = str(c).strip()
        if c and c not in uniq:
            uniq.append(c)
        if len(uniq) >= 15:
            break
    breadth_ma50_cnt = 0
    breadth_ma20_cnt = 0
    rsi_vals = []
    total = 0
    for code in uniq:
        try:
            d = _yahoo_ohlc_df(code, interval="1d", y_range="6mo")
        except httpx.HTTPError:
            continue
        if d.empty or len(d) < 30:
            continue
        c = d["close"].astype(float)
        ma50 = c.rolling(50, min_periods=1).mean()
        ma20 = c.rolling(20, min_periods=1).mean()
        if c.iloc[-1] > ma50.iloc[-1]:
            breadth_ma50_cnt += 1
        if c.iloc[-1] > ma20.iloc[-1]:
            breadth_ma20_cnt += 1
        rsi_vals.append(float(_rsi(c, 14).iloc[-1]))
        total += 1
    pct_ma50 = float(breadth_ma50_cnt / total) if total else 0.0
    pct_ma20 = float(breadth_ma20_cnt / total) if total else 0.0
    avg_rsi = float(np.mean(rsi_vals)) if rsi_vals else 0.0
    if pct_ma50 > 0.6 and avg_rsi > 55:
        env_label = "Risk-On"
    elif pct_ma50 < 0.4 and avg_rsi < 45:
        env_label = "Risk-Off"
    else:
        env_label = "中性"

    return {
        "symbol": sym,
        "symbol_sentiment": {
            "rsi14": float(rsi14),
            "slope_norm": slope_norm,
            "vol": vol,
            "score": score,
            "label": sym_label,
        },
        "environment_sentiment": {
            "universe_size": total,
            "pct_above_ma50": pct_ma50,
            "pct_above_ma20": pct_ma20,
            "avg_rsi": avg_rsi,
            "label": env_label,
        }
    }

# --- News summary ---
class NewsRequest(BaseModel):
    symbol: str
    universe: Optional[List[str]] = None

def _yahoo_quote_name(symbol: str) -> Optional[str]:
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinLabBot/1.0)", "Accept": "application/json"}
    try:
        with httpx.Client(timeout=httpx.Timeout(6.0, read=8.0)) as client:
            r = client.get(url, params={"symbols": symbol}, headers=headers)
            r.raise_for_status()
            j = r.json()
        result = ((j or {}).get("quoteResponse") or {}).get("result") or []
        if not result:
            return None
        q = result[0]
        name = q.get("shortName") or q.get("longName") or q.get("displayName")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        return None
    return None

def _yahoo_news_search(query: str, news_count: int = 6, lang: str = "en-US", region: str = "US", *, require_keywords: Optional[List[str]] = None) -> List[dict]:
    base_urls = [
        "https://query1.finance.yahoo.com/v1/finance/search",
        "https://query2.finance.yahoo.com/v1/finance/search",
    ]
    params = {
        "q": query,
        "newsCount": str(int(news_count)),
        "quotesCount": "0",
        "lang": lang,
        "region": region,
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinLabBot/1.0)", "Accept": "application/json"}
    last_err = None
    for url in base_urls:
        try:
            with httpx.Client(timeout=httpx.Timeout(8.0, read=12.0)) as client:
                r = client.get(url, params=params, headers=headers)
                r.raise_for_status()
                j = r.json()
                news = (j or {}).get("news") or []
                out = []
                for n in news:
                    title = n.get("title")
                    link = n.get("link") or n.get("linkUrl")
                    pub = n.get("publisher") or n.get("provider")
                    ts = n.get("pubDate") or n.get("published_at")
                    summ = n.get("summary") or n.get("body") or ""
                    text_blob = " ".join([str(x) for x in [title, summ, pub] if x])
                    if require_keywords:
                        hit = False
                        t = text_blob.lower()
                        for kw in require_keywords:
                            if not kw:
                                continue
                            if kw.lower() in t:
                                hit = True
                                break
                        if not hit:
                            continue
                    # Yahoo returns millis
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(float(ts)/1000.0, tz=timezone.utc).isoformat()
                    else:
                        dt = None
                    if title and link:
                        out.append({"title": title, "url": link, "publisher": pub, "published_at": dt})
                if out:
                    return out[:news_count]
        except Exception as e:
            last_err = e
            continue
    # If all fail, return empty list
    return []

def _gdelt_news_search(
    query: str,
    news_count: int = 8,
    *,
    lang_hint: str = "english",
    require_keywords: Optional[List[str]] = None,
    timespan: str = "14d",
    exclude_domains: Optional[List[str]] = None,
    country_hint: Optional[str] = None,
) -> List[dict]:
    """Query GDELT Doc API for recent articles.
    lang_hint: 'english' or 'chinese' (case-insensitive) → used in sourcelang filter.
    timespan: like '7d','14d','30d'.
    country_hint: optional sourcecountry code like 'TW' to prioritize local language sources.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    # Normalize language token for sourcelang (lowercase required by GDELT)
    lang_token = None
    if lang_hint:
        if lang_hint.lower().startswith("zh") or lang_hint.lower().startswith("chi") or lang_hint.lower()=="chinese":
            lang_token = "chinese"
        elif lang_hint.lower()=="english" or lang_hint.lower().startswith("en"):
            lang_token = "english"
    lang_filter = f" sourcelang:{lang_token}" if lang_token else ""
    country_filter = f" sourcecountry:{country_hint}" if country_hint else ""

    q = f"({query}){lang_filter}{country_filter}"
    params = {
        "query": q,
        "format": "json",
        "timespan": timespan,
        "sort": "DateDesc",
        "maxrecords": str(news_count * 10),  # fetch extra, we'll filter
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinLabBot/1.0)", "Accept": "application/json"}
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0, read=15.0)) as client:
            r = client.get(url, params=params, headers=headers)
            r.raise_for_status()
            j = r.json()
    except Exception:
        return []
    arts = (j or {}).get("articles") or []
    out: List[dict] = []
    for a in arts:
        title = a.get("title")
        link = a.get("url")
        pub = a.get("sourceCommonName") or a.get("domain") or a.get("sourcecountry")
        ts = a.get("seendate") or a.get("publishdate")
        text_blob = " ".join([str(x) for x in [title, pub, a.get("url") ] if x])
        if require_keywords:
            t = text_blob.lower()
            if not any((kw and kw.lower() in t) for kw in require_keywords):
                continue
        if exclude_domains and link:
            try:
                host = urlparse(link).netloc.lower()
                if any(dom in host for dom in exclude_domains):
                    continue
            except Exception:
                pass
        # GDELT seendate often like '20240902130500' or ISO
        dt = None
        if isinstance(ts, str):
            try:
                if "T" in ts:
                    # assume ISO
                    dt = ts if ts.endswith("Z") else datetime.fromisoformat(ts).astimezone(timezone.utc).isoformat()
                else:
                    dt = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat()
            except Exception:
                dt = None
        if title and link:
            out.append({"title": title, "url": link, "publisher": pub, "published_at": dt})
    return out[:news_count]

@app.post("/news/summary")
def news_summary(req: NewsRequest):
    sym = req.symbol
    # Pick market query based on symbol suffix
    market_query = "stock market"
    lang = "en-US"; region = "US"
    if sym.upper().endswith(".TW"):
        market_query = "台股 OR 臺股 OR 台灣 股市"
        lang = "zh-TW"; region = "TW"
    # Build a richer query for the symbol to improve relevance
    sym_name = _yahoo_quote_name(sym)
    no_suf = sym.split(".")[0] if "." in sym else sym
    q_parts = [sym]
    if sym_name:
        q_parts.append(f'"{sym_name}"')
    if no_suf and no_suf != sym:
        q_parts.append(no_suf)
    sym_query = " OR ".join(q_parts + ["shares", "stock"]) if lang.startswith("en") else " OR ".join(q_parts)
    # Build strict keyword filters to ensure relevance
    keywords = [sym.lower()]
    if no_suf:
        keywords.append(no_suf.lower())
    if sym_name:
        keywords.extend([sym_name.lower()])
        # split words for English names
        keywords.extend([w.lower() for w in re.split(r"\W+", sym_name) if len(w) > 2])
    # Chinese helpers
    if sym.upper().endswith('.TW'):
        keywords.extend(['股價','股票','公司','台積','台股','臺股'])

    # Prefer GDELT; query both English/Chinese for TW市場
    is_tw = sym.upper().endswith(".TW")
    exclude = ["yahoo.com", "finance.yahoo.com", "news.yahoo.com", "tw.news.yahoo.com", "tw.stock.yahoo.com"]

    def dedupe(items: List[dict]) -> List[dict]:
        seen = set()
        out = []
        for it in items:
            u = (it.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(it)
        return out

    sym_list: List[dict] = []
    mkt_list: List[dict] = []

    # Try OpenBB first if available
    if os.getenv("NEWS_PROVIDER", "openbb").lower() == "openbb":
        sym_list = _openbb_news_search(sym) or []
        # market query via search
        mkt_list = _openbb_news_search(market_query) or []

    # Then try GDELT English (priority)
    sym_list += _gdelt_news_search(
        sym_query, news_count=8, lang_hint="english", require_keywords=keywords,
        exclude_domains=exclude, timespan="30d", country_hint="US")
    mkt_list += _gdelt_news_search(
        market_query, news_count=6, lang_hint="english",
        exclude_domains=exclude, timespan="30d", country_hint="US")
    # Try GDELT Chinese for TW
    if is_tw:
        sym_list += _gdelt_news_search(
            sym_query, news_count=8, lang_hint="chinese", require_keywords=keywords,
            exclude_domains=exclude, timespan="30d", country_hint="TW")
        mkt_list += _gdelt_news_search(
            market_query, news_count=6, lang_hint="chinese",
            exclude_domains=exclude, timespan="30d", country_hint="TW")

    sym_list = dedupe(sym_list)
    mkt_list = dedupe(mkt_list)

    # If still empty, relax filters progressively
    if not sym_list:
        # 1) GDELT without language filter (wider net)
        sym_list += _gdelt_news_search(sym_query, news_count=8, lang_hint=None, timespan="60d")
    if not mkt_list:
        mkt_list += _gdelt_news_search(market_query, news_count=6, lang_hint=None, timespan="60d")

    sym_list = dedupe(sym_list)
    mkt_list = dedupe(mkt_list)

    # Final safety fallback to Yahoo finance search if still nothing
    if not sym_list:
        sym_list = _yahoo_news_search(sym_query, news_count=8, lang=lang, region=region, require_keywords=keywords)
    if not mkt_list:
        mkt_list = _yahoo_news_search(market_query, news_count=6, lang=lang, region=region)

    return {"symbol": sym, "symbol_news": sym_list[:8], "market_news": mkt_list[:6]}

# --- OpenBB news (optional) ---
def _openbb_news_search(query: str, limit: int = 8) -> List[dict]:
    try:
        try:
            # OpenBB SDK v4
            from openbb import ob as _ob
        except Exception:
            try:
                from openbb import obb as _ob  # alt alias
            except Exception:
                _ob = None
        if _ob is None:
            return []

        items = []
        news_mod = getattr(_ob, "news", None)
        tried = []
        for method_name in ("search", "company", "query"):
            if not news_mod or not hasattr(news_mod, method_name):
                continue
            func = getattr(news_mod, method_name)
            try:
                if method_name == "company":
                    resp = func(symbol=query, limit=limit)
                else:
                    resp = func(query=query, limit=limit)
            except Exception:
                tried.append(method_name)
                continue
            # Normalize response
            data = None
            if resp is None:
                continue
            if hasattr(resp, "results"):
                data = resp.results
            elif hasattr(resp, "to_df"):
                try:
                    df = resp.to_df()
                    data = df.to_dict(orient="records")
                except Exception:
                    data = None
            elif isinstance(resp, dict) and "data" in resp:
                data = resp["data"]
            elif isinstance(resp, (list, tuple)):
                data = resp
            if not data:
                continue
            out: List[dict] = []
            for it in data:
                if isinstance(it, dict):
                    title = it.get("title") or it.get("headline") or it.get("name")
                    url = it.get("url") or it.get("link")
                    src = it.get("source") or it.get("publisher")
                    ts = it.get("published") or it.get("published_at") or it.get("date")
                else:
                    title = str(it)
                    url = None
                    src = None
                    ts = None
                if not title:
                    continue
                iso = None
                if isinstance(ts, str):
                    try:
                        if "T" in ts:
                            iso = ts if ts.endswith("Z") else datetime.fromisoformat(ts).astimezone(timezone.utc).isoformat()
                        else:
                            iso = datetime.fromisoformat(ts).astimezone(timezone.utc).isoformat()
                    except Exception:
                        iso = None
                elif isinstance(ts, (int, float)):
                    try:
                        iso = datetime.fromtimestamp(float(ts)/1000.0, tz=timezone.utc).isoformat()
                    except Exception:
                        iso = None
                out.append({"title": title, "url": url, "publisher": src, "published_at": iso})
            if out:
                return out[:limit]
        return []
    except Exception:
        return []


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
