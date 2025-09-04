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

        # Prefer Yahoo daily data to match frontend chart; fallback to DB if Yahoo unavailable
        full = _yahoo_close_df(symbol, interval="1d", y_range="10y")
        if not full.empty:
            start_utc = pd.to_datetime(start, utc=True)
            end_utc = pd.to_datetime(end, utc=True)
            q_df = full[(full.index >= start_utc) & (full.index <= end_utc)]
        else:
            q_df = pd.DataFrame()
        if q_df.empty:
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
        if q_df.empty:
            raise ValueError("No data in range")
        fast_ma = sma(q_df["close"], fast)
        slow_ma = sma(q_df["close"], slow)
        # detect true cross points（上一根不在同側才算交叉）
        cross_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        cross_dn = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        # 建立部位：起始空手；金叉=1、死叉=0，其他沿用前一根
        pos_series = pd.Series(np.nan, index=fast_ma.index, dtype=float)
        pos_series[cross_up.fillna(False)] = 1
        pos_series[cross_dn.fillna(False)] = 0
        if len(pos_series) > 0:
            pos_series.iloc[0] = 0
        pos = pos_series.ffill().fillna(0).astype(int)

        ret = np.log(q_df["close"]).diff().fillna(0.0)
        strat_ret = ret * pos.shift(1).fillna(0.0)
        equity = (1 + strat_ret).cumprod() * cash

        # Events based on position transitions; execute at signal-bar close
        closes = q_df["close"].astype(float)
        idx = list(q_df.index)
        events = []
        trades_detail = []
        for i in range(1, len(idx)):
            prev = int(pos.iloc[i-1])
            cur = int(pos.iloc[i])
            if prev == 0 and cur == 1:
                t = idx[i]
                events.append({"ts": t.isoformat(), "type": "buy", "price": float(closes.iloc[i])})
            elif prev == 1 and cur == 0:
                t = idx[i]
                events.append({"ts": t.isoformat(), "type": "sell", "price": float(closes.iloc[i])})

        # Pair into trades
        open_trade = None
        for ev in events:
            if ev["type"] == "buy" and open_trade is None:
                open_trade = {"entry_ts": ev["ts"], "entry_price": ev["price"]}
            elif ev["type"] == "sell" and open_trade is not None:
                exit_ts = ev["ts"]
                exit_price = ev["price"]
                entry_price = float(open_trade["entry_price"])
                ret_pct = (exit_price / entry_price) - 1.0
                trades_detail.append({
                    "entry_ts": open_trade["entry_ts"],
                    "entry_price": entry_price,
                    "exit_ts": exit_ts,
                    "exit_price": exit_price,
                    "return": ret_pct,
                })
                open_trade = None

        report = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "params": {"fast": fast, "slow": slow},
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(equity.iloc[-1]/cash - 1.0),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "trades": int(len(trades_detail)),
            "events": events,
            "trades_detail": trades_detail,
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
