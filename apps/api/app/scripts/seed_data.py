import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from ..db import engine, Base
from ..models import Symbol

def ensure_timescale(conn):
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
    except Exception as e:
        print("Timescale extension note:", e)

def ensure_tables():
    # Recreate ohlcv with composite PK to satisfy Timescale hypertable rules.
    # Safe in dev: drop only the ohlcv table before re-creating.
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS ohlcv"))
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        try:
            conn.execute(text("SELECT create_hypertable('ohlcv','ts', if_not_exists => TRUE);"))
        except Exception as e:
            print("Hypertable note:", e)

def gen_gbm(n=800, s0=100, mu=0.08, sigma=0.25, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1/252
    shocks = rng.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), n)
    log_prices = np.cumsum(shocks) + np.log(s0)
    prices = np.exp(log_prices)
    close = prices
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.normal(0.001, 0.005, n).clip(-0.01, 0.02))
    low  = np.minimum(open_, close) * (1 - rng.normal(0.001, 0.005, n).clip(-0.02, 0.01))
    vol  = rng.integers(1e5, 5e5, n)
    return pd.DataFrame({"open":open_, "high":high, "low":low, "close":close, "volume":vol})

def seed_symbols_and_data():
    from ..db import engine
    with Session(bind=engine) as session:
        codes = [("TW","2330.TW","台積電"), ("US","AAPL","Apple Inc.")]
        code_to_id = {}
        for mkt, code, name in codes:
            s = session.query(Symbol).filter_by(code=code).first()
            if not s:
                s = Symbol(market=mkt, code=code, name=name)
                session.add(s)
                session.flush()
            code_to_id[code] = s.id
        # Ensure symbol rows are committed before inserting into ohlcv via a separate connection
        session.commit()

        start = datetime(2022,1,3)
        days = 820
        dates = []
        d = start
        while len(dates) < days:
            if d.weekday() < 5:
                dates.append(d)
            d += timedelta(days=1)

        for code in code_to_id:
            df = gen_gbm(n=len(dates), s0=100 if code=="AAPL" else 500, seed=42 if code=="AAPL" else 7)
            df["ts"] = dates
            df["symbol_id"] = code_to_id[code]
            session.execute(text("DELETE FROM ohlcv WHERE symbol_id = :sid"), {"sid": code_to_id[code]})
            # Use engine (separate connection) after committing symbols to avoid FK violations
            df.to_sql("ohlcv", session.bind, if_exists="append", index=False, method=None)
        session.commit()

if __name__ == "__main__":
    print("Initializing Timescale tables and seeding data...")
    with engine.connect() as conn:
        ensure_timescale(conn)
    ensure_tables()
    seed_symbols_and_data()
    print("Done.")
