import enum
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import ForeignKey, String, Integer, DateTime, Enum, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class CategoryEnum(str, enum.Enum):
    STATIONARY = "STATIONARY"
    COMPONENT = "COMPONENT"

class ItemTypeEnum(str, enum.Enum):
    SINGLE_COMPONENT = "SINGLE_COMPONENT"
    FULL_BUILD = "FULL_BUILD"
    BUNDLE = "BUNDLE"
    UNKNOWN = "UNKNOWN"

class StatusEnum(str, enum.Enum):
    ACTIVE = "ACTIVE"
    SOLD = "SOLD"
    IGNORED = "IGNORED"


class ComponentTypeEnum(str, enum.Enum):
    CPU = "CPU"
    GPU = "GPU"
    RAM = "RAM"
    MOTHERBOARD = "MOTHERBOARD"
    PSU = "PSU"
    STORAGE = "STORAGE"
    CASE = "CASE"
    OTHER = "OTHER"


class Base(DeclarativeBase):
    pass


class Ad(Base):
    __tablename__ = "ads"

    id: Mapped[int] = mapped_column(primary_key=True)
    blocket_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    url: Mapped[str] = mapped_column(String, unique=True, index=True)
    title: Mapped[str] = mapped_column(String)
    raw_description: Mapped[str] = mapped_column(Text)
    price_sek: Mapped[int] = mapped_column(Integer)

    main_image_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    item_type: Mapped[ItemTypeEnum] = mapped_column(Enum(ItemTypeEnum), default=ItemTypeEnum.UNKNOWN)
    
    category: Mapped[CategoryEnum] = mapped_column(Enum(CategoryEnum))
    status: Mapped[StatusEnum] = mapped_column(Enum(StatusEnum), default=StatusEnum.ACTIVE)
    
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships (Cascade Delete to ensure cleanups)
    ai_evaluation: Mapped[Optional["AiEvaluation"]] = relationship(
        back_populates="ad", cascade="all, delete-orphan"
    )
    extracted_components: Mapped[List["ExtractedComponent"]] = relationship(
        back_populates="ad", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Ad(blocket_id={self.blocket_id}, title={self.title}, price={self.price_sek})>"


class AiEvaluation(Base):
    __tablename__ = "ai_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id", ondelete="CASCADE"), unique=True)
    deal_score: Mapped[int] = mapped_column(Integer)
    estimated_market_value_sek: Mapped[int] = mapped_column(Integer)
    ai_reasoning: Mapped[str] = mapped_column(Text)
    
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationship back to Ad
    ad: Mapped["Ad"] = relationship(back_populates="ai_evaluation")


class ExtractedComponent(Base):
    __tablename__ = "extracted_components"

    id: Mapped[int] = mapped_column(primary_key=True)
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id", ondelete="CASCADE"))
    component_type: Mapped[ComponentTypeEnum] = mapped_column(Enum(ComponentTypeEnum))
    component_name: Mapped[str] = mapped_column(String)
    estimated_individual_value: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationship back to Ad
    ad: Mapped["Ad"] = relationship(back_populates="extracted_components")
