from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, UniqueConstraint, Index, Identity, PrimaryKeyConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    tz = Column(String, default="Asia/Taipei")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UserAuth(Base):
    __tablename__ = "user_auth"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)

class Symbol(Base):
    __tablename__ = "symbols"
    id = Column(Integer, primary_key=True, index=True)
    market = Column(String, nullable=False, default="TW")
    code = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False, default="")
    sector = Column(String, nullable=True)

class OHLCV(Base):
    __tablename__ = "ohlcv"
    # Use composite primary key (symbol_id, ts) to satisfy TimescaleDB's
    # requirement that unique constraints include the partition key.
    symbol_id = Column(Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False)
    ts = Column(DateTime(timezone=False), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('symbol_id', 'ts', name='pk_ohlcv_symbol_ts'),
        Index("idx_ohlcv_symbol_ts", "symbol_id", "ts"),
    )

class BacktestRun(Base):
    __tablename__ = "bt_runs"
    id = Column(String, primary_key=True)  # rq job id
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    status = Column(String, default="queued")
    cfg_json = Column(JSON, nullable=False)
    report_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
