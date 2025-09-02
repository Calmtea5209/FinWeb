import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://finlab:finlab@localhost:5432/finlab")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Optional session provider: yields a Session or None if connection fails
def get_db_safe():
    try:
        db = SessionLocal()
    except Exception:
        db = None
    try:
        yield db
    finally:
        try:
            if db is not None:
                db.close()
        except Exception:
            pass
