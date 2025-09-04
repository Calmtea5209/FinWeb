import os, json, math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import httpx
from rq import Queue, get_current_job
from redis import Redis
from sqlalchemy.orm import Session
from .db import SessionLocal
from .models import BacktestRun, Symbol
from .indicators import sma

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = Redis.from_url(REDIS_URL)
q = Queue("default", connection=redis)

def _yahoo_close_df(symbol: str, interval: str = "1d", y_range: str = "10y") -> pd.DataFrame:
    """Fetch Yahoo Finance OHLC and return a tz-aware (UTC) close series DataFrame.
    Returns DataFrame indexed by UTC timestamps with a 'close' column.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinLabBot/1.0)", "Accept": "application/json"}
    params = {"interval": interval, "range": y_range, "includePrePost": "false", "events": "div,splits"}
    with httpx.Client(timeout=httpx.Timeout(10.0, read=20.0)) as client:
        r = client.get(url, params=params, headers=headers)
        r.raise_for_status()
        j = r.json()
    result = ((j or {}).get("chart") or {}).get("result") or []
    if not result:
        return pd.DataFrame(columns=["ts","close"]).set_index("ts")
    res = result[0]
    ts = res.get("timestamp") or []
    ind = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    close_a = ind.get("close") or []
    rows = []
    for i in range(min(len(ts), len(close_a))):
        c = close_a[i]
        if c is None:
            continue
        try:
            cf = float(c)
        except Exception:
            continue
        if not math.isfinite(cf):
            continue
        rows.append({"ts": datetime.fromtimestamp(ts[i], tz=timezone.utc), "close": cf})
    if not rows:
        return pd.DataFrame(columns=["ts","close"]).set_index("ts")
    return pd.DataFrame(rows).sort_values("ts").set_index("ts")

def run_backtest_job(cfg_json: dict):
    """最小 SMA Cross 回測：快線 > 慢線 進場；反之出場。

    若 DB 資料不足，回落至 Yahoo Finance 日資料；允許 DB 無該代碼。
    """
    session: Session = SessionLocal()
    try:
        cfg = cfg_json
        symbol = cfg["symbol"]
        start = cfg["start"]
        end = cfg["end"]
        fast = int(cfg["params"].get("fast", 10))
        slow = int(cfg["params"].get("slow", 30))
        cash = float(cfg.get("cash", 1_000_000))

        # Try DB first if symbol exists
        q_df = pd.DataFrame()
        try:
            sym = session.query(Symbol).filter_by(code=symbol).first()
        except Exception:
            sym = None
        if sym is not None:
            q_df = pd.read_sql(
                f"""SELECT ts, close FROM ohlcv WHERE symbol_id={sym.id}
                     AND ts >= %(s)s AND ts <= %(e)s ORDER BY ts""" ,
                session.bind,
                params={"s": start, "e": end},
                parse_dates=["ts"]
            )
            if not q_df.empty:
                q_df.set_index("ts", inplace=True)

        # Fallback to Yahoo if DB had no rows
        if q_df.empty:
            full = _yahoo_close_df(symbol, interval="1d", y_range="10y")
            if full.empty:
                raise ValueError("No data in range")
            start_utc = pd.to_datetime(start, utc=True)
            end_utc = pd.to_datetime(end, utc=True)
            q_df = full[(full.index >= start_utc) & (full.index <= end_utc)]
            if q_df.empty:
                raise ValueError("No data in range")
        fast_ma = sma(q_df["close"], fast)
        slow_ma = sma(q_df["close"], slow)
        signal = (fast_ma > slow_ma).astype(int)
        ret = np.log(q_df["close"]).diff().fillna(0.0)
        strat_ret = ret * signal.shift(1).fillna(0.0)
        equity = (1 + strat_ret).cumprod() * cash

        report = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "params": {"fast": fast, "slow": slow},
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(equity.iloc[-1]/cash - 1.0),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "trades": int((signal.diff().abs() == 1).sum() // 2),
        }

        jid = get_current_job().id if get_current_job() else None
        if jid:
            run = session.get(BacktestRun, jid)
            if run:
                run.status = "finished"
                run.report_json = report
                session.commit()
    except Exception as e:
        jid = get_current_job().id if get_current_job() else None
        if jid:
            run = session.get(BacktestRun, jid)
            if run:
                run.status = "failed"
                run.report_json = {"error": str(e)}
                session.commit()
    finally:
        session.close()

def enqueue_backtest(cfg_json: dict) -> str:
    job = q.enqueue(run_backtest_job, args=(cfg_json,))
    # 建立 DB 紀錄
    session = SessionLocal()
    try:
        br = BacktestRun(id=job.id, status="queued", cfg_json=cfg_json)
        session.add(br)
        session.commit()
    finally:
        session.close()
    return job.id
