"""Database schema.

Two schema decisions carry most of the design weight:

**No personal data has a column.** There is no seller name, no phone, no email,
no postal code and no coordinates anywhere in this file. `sources/` refuses to
ingest them; this module refuses to store them. A leak would need both layers to
be wrong at once.

**`raw_description` is nullable on purpose.** It is the only free-text field we
keep, it is the field most likely to contain a seller's phone number typed into
the body, and it is deleted once extraction has read it (see the retention pass
in `pipeline/janitor.py`). A schema that *requires* the raw text cannot express
that, so it does not require it.
"""

import enum
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
    """One listing, from any source.

    Identity is `(source, external_id)`, not a Blocket id — the `Source`
    protocol exists so a second marketplace can be added without a migration,
    and a primary key naming one vendor would undo that.
    """

    __tablename__ = "ads"
    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_ads_source_external_id"),
        # The AI phase scans for "active, not yet evaluated at this content
        # hash". Without this it is a full table scan on every run.
        Index("ix_ads_status_content_hash", "status", "content_hash"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String, default="blocket", index=True)
    external_id: Mapped[str] = mapped_column(String, index=True)
    url: Mapped[str] = mapped_column(String, unique=True, index=True)

    title: Mapped[str] = mapped_column(String)
    # Nullable by design — dropped after extraction. See the module docstring.
    raw_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    price_sek: Mapped[int] = mapped_column(Integer)

    # Coarse location only. Never a postal code, never coordinates.
    city: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Blocket publishes this as a structured attribute, so it costs no LLM call.
    condition: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    image_count: Mapped[int] = mapped_column(Integer, default=0)

    # sha256 of (title, description, price). The AI phase skips an ad whose hash
    # matches its last evaluation, so a re-scrape that changed nothing costs no
    # quota. The old `~Ad.ai_evaluation.has()` check could not express this: it
    # re-analysed nothing, even after a seller cut the price.
    content_hash: Mapped[str] = mapped_column(String(64), index=True)

    item_type: Mapped[ItemTypeEnum] = mapped_column(
        Enum(ItemTypeEnum), default=ItemTypeEnum.UNKNOWN
    )
    category: Mapped[CategoryEnum] = mapped_column(Enum(CategoryEnum))
    status: Mapped[StatusEnum] = mapped_column(Enum(StatusEnum), default=StatusEnum.ACTIVE)

    listed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    # When raw_description was dropped. Non-null means extraction already ran and
    # the text is gone for good — distinguishes "discarded" from "never had one".
    description_dropped_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    ai_evaluation: Mapped[Optional["AiEvaluation"]] = relationship(
        back_populates="ad", cascade="all, delete-orphan"
    )
    extracted_components: Mapped[List["ExtractedComponent"]] = relationship(
        back_populates="ad", cascade="all, delete-orphan"
    )
    ai_failure: Mapped[Optional["AiFailure"]] = relationship(
        back_populates="ad", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Ad({self.source}:{self.external_id} {self.price_sek} kr {self.title!r})>"


class AiEvaluation(Base):
    __tablename__ = "ai_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id", ondelete="CASCADE"), unique=True)

    item_type: Mapped[ItemTypeEnum] = mapped_column(Enum(ItemTypeEnum))
    # 0.0–1.0. Multiplies the deterministic price delta when ranking, so the AI
    # can discount a mathematically good deal it distrusts without inventing a
    # price of its own.
    confidence: Mapped[float] = mapped_column(default=1.0)
    # Newline-separated short flags ("ej testad", "no receipt"). Rendered as
    # chips. Kept as text rather than a table because they are display-only and
    # never queried.
    risk_flags: Mapped[str] = mapped_column(Text, default="")
    ai_reasoning: Mapped[str] = mapped_column(Text)

    # The hash of the ad content this verdict was formed from. If the ad changes,
    # the hash stops matching and the ad becomes eligible for re-evaluation.
    content_hash: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String, default="")
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    ad: Mapped["Ad"] = relationship(back_populates="ai_evaluation")

    @property
    def flags(self) -> list[str]:
        return [line for line in self.risk_flags.splitlines() if line.strip()]


class AiFailure(Base):
    """Dead-letter row: an ad the AI phase could not evaluate.

    This table is the fix for the bug that mattered most. Previously a batch
    that failed schema parsing hit `continue`, and those ads were never
    retried and never counted — 474 ads scraped, 120 evaluated, and nothing
    anywhere said why. Recording the failure makes the gap queryable, bounds
    retries, and stops a permanently malformed ad from being retried forever.
    """

    __tablename__ = "ai_failures"

    id: Mapped[int] = mapped_column(primary_key=True)
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id", ondelete="CASCADE"), unique=True)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    last_error: Mapped[str] = mapped_column(Text)
    last_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    ad: Mapped["Ad"] = relationship(back_populates="ai_failure")


class PriceObservation(Base):
    """One sighting of one part at one price. Append-only.

    ### The table the retention design rests on

    Note what is *absent*: no foreign key to `ads`, no ad id, no url, no seller,
    nothing personal. That is the entire point. Because an observation does not
    reference the listing it came from, **deleting the ad leaves the data point
    intact** — the janitor can purge aggressively for privacy while the price
    history keeps compounding for accuracy.

    Those two goals normally pull against each other. Here one table satisfies
    both, which is the main reason the schema is shaped this way.

    Immutable by convention: rows are inserted, never updated. A price is a fact
    about a moment, and editing history would make the median unreproducible.
    """

    __tablename__ = "price_observations"
    __table_args__ = (
        # The index query is always "this part, recent first".
        Index("ix_price_obs_part_time", "canonical_name", "observed_at"),
        # Idempotency: re-scraping an unchanged ad must not stack duplicate
        # observations and fake a deeper sample than really exists.
        UniqueConstraint(
            "canonical_name", "source", "source_ref", "price_sek",
            name="uq_price_obs_dedup",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String, index=True)
    component_type: Mapped[ComponentTypeEnum] = mapped_column(Enum(ComponentTypeEnum))
    price_sek: Mapped[int] = mapped_column(Integer)
    condition: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    source: Mapped[str] = mapped_column(String, default="blocket")
    # An opaque marketplace id, kept *only* to deduplicate re-scrapes. It is not
    # a foreign key and is never resolved back to an ad — the ad may be long
    # gone. Deliberately not a link.
    source_ref: Mapped[str] = mapped_column(String)

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )


class ExtractedComponent(Base):
    __tablename__ = "extracted_components"

    id: Mapped[int] = mapped_column(primary_key=True)
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id", ondelete="CASCADE"), index=True)
    component_type: Mapped[ComponentTypeEnum] = mapped_column(Enum(ComponentTypeEnum))
    component_name: Mapped[str] = mapped_column(String)
    # Normalized form used for matching ("rtx 3080" for "RTX 3080 Asus TUF").
    # Phase 4 replaces the normalizer; the column it writes to is already here.
    canonical_name: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)

    ad: Mapped["Ad"] = relationship(back_populates="extracted_components")
