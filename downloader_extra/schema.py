from pydantic import BaseModel
from sqlalchemy import Column, String, Float, Integer
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class MacroIndicator(Base):
    __tablename__ = "indicators"

    economy = Column(String, primary_key=True, nullable=False)
    year = Column(Integer, primary_key=True, nullable=False)
    value = Column(Float, nullable=True)
    indicator_id = Column(String, primary_key=True, nullable=False)
    db_id = Column(Integer, primary_key=True, nullable=False, index=True)


class IngestResponse(BaseModel):
    indicator_id: str
    db_id: int
    rows_inserted: int
    status: str


class IngestRequest(BaseModel):
    indicator_id: str
    db_id: int
