"""Database models for pocket-nutrition-ingest."""

import os

from sqlalchemy import Boolean, Column, Float, Integer, String, Text, UniqueConstraint, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://pocket_nutrition:pocket_nutrition@postgresql:5432/pocket_nutrition")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

NUTRIENT_COLUMNS = [
    "energy_kcal", "water_g", "protein_g", "fat_g", "carbohydrates_g",
    "fiber_g", "sugars_g", "calcium_mg", "iron_mg", "magnesium_mg",
    "sodium_mg", "vitamin_c_mg",
    "saturated_fat_g", "monounsaturated_fat_g", "polyunsaturated_fat_g",
    "cholesterol_mg", "potassium_mg", "phosphorus_mg", "zinc_mg",
    "vitamin_a_ug", "vitamin_d_ug", "vitamin_b6_mg", "vitamin_b12_ug",
    "niacin_mg",
]


class Base(DeclarativeBase):
    pass


class TrainingPair(Base):
    __tablename__ = "training_pairs"
    __table_args__ = (
        UniqueConstraint("source", "food_name", "cooking_method", name="uq_training_pair"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(16), nullable=False)
    food_name = Column(Text, nullable=False)
    cooking_method = Column(String(32), nullable=False)
    raw_profile = Column(Text, nullable=False)    # JSON per 100g
    cooked_profile = Column(Text, nullable=False)  # JSON per 100g raw (yield-normalised)
    food_group = Column(String(128), nullable=True)
    yield_f = Column(Float, nullable=True)
    is_theoretical = Column(Boolean, nullable=False, default=False)


def init_training_tables() -> None:
    """Drop and recreate training_pairs."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS training_pairs CASCADE"))
    Base.metadata.create_all(engine, tables=[TrainingPair.__table__])


def get_session() -> Session:
    return SessionLocal()
