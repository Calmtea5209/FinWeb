from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class IndicatorRequest(BaseModel):
    symbol: str
    limit: int = 200
    indicators: List[str] = Field(default_factory=lambda: ["sma:20","rsi:14"])

class SimilarSearchRequest(BaseModel):
    symbol: str
    start: str
    end: str
    m: int = 30
    top: int = 10
    universe: Optional[List[str]] = None

class BacktestConfig(BaseModel):
    symbol: str
    start: str
    end: str
    cash: float = 1_000_000
    strategy: str = "sma_cross"
    params: Dict[str, float] = Field(default_factory=lambda: {"fast": 10, "slow": 30})

class BacktestSubmitRequest(BaseModel):
    config: BacktestConfig
    user_id: Optional[int] = None
