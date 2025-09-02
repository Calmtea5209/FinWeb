import os, json
import numpy as np
import pandas as pd
from datetime import datetime
from rq import Queue, get_current_job
from redis import Redis
from sqlalchemy.orm import Session
from .db import SessionLocal
from .models import BacktestRun, Symbol
from .indicators import sma

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = Redis.from_url(REDIS_URL)
q = Queue("default", connection=redis)

def run_backtest_job(cfg_json: dict):
    """最小 SMA Cross 回測：快線 > 慢線 進場；反之出場。"""
    session: Session = SessionLocal()
    try:
        cfg = cfg_json
        symbol = cfg["symbol"]
        start = cfg["start"]
        end = cfg["end"]
        fast = int(cfg["params"].get("fast", 10))
        slow = int(cfg["params"].get("slow", 30))
        cash = float(cfg.get("cash", 1_000_000))

        sym = session.query(Symbol).filter_by(code=symbol).first()
        if not sym:
            raise ValueError(f"Unknown symbol {symbol}")
        q_df = pd.read_sql(
            f"""SELECT ts, close FROM ohlcv WHERE symbol_id={sym.id}
                 AND ts >= %(s)s AND ts <= %(e)s ORDER BY ts""" ,
            session.bind,
            params={"s": start, "e": end},
            parse_dates=["ts"]
        )
        if q_df.empty:
            raise ValueError("No data in range")

        q_df.set_index("ts", inplace=True)
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
