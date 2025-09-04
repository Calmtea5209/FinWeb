from pydantic import BaseModel, Field, EmailStr
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
    # Only consider candidate matches within the last N years (relative to now)
    lookback_years: Optional[int] = 5

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


# --- Auth ---
class UserOut(BaseModel):
    id: int
    email: EmailStr
    tz: Optional[str] = "Asia/Taipei"


class AuthRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    tz: Optional[str] = "Asia/Taipei"


class AuthLogin(BaseModel):
    email: EmailStr
    password: str


class AuthToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: Optional[str] = None  # ISO
    user: UserOut
