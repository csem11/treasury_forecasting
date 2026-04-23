"""Schema and anomaly validation for Treasury and FRED datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_CURVE_COLUMNS = [
    "1_mo", "3_mo", "6_mo", "1_yr", "2_yr", "3_yr",
    "5_yr", "7_yr", "10_yr", "20_yr", "30_yr",
]

# Reasonable yield bounds in percentage points.
YIELD_MIN = 0.0
YIELD_MAX = 25.0

# Alert if any maturity has more than this fraction of NaNs.
MAX_NULL_FRACTION = 0.05

# Alert if 10Y moves more than this many bps in one day.
MAX_DAILY_MOVE_BPS = 100.0


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def log_all(self) -> None:
        for msg in self.errors:
            log.error("VALIDATION ERROR: %s", msg)
        for msg in self.warnings:
            log.warning("VALIDATION WARNING: %s", msg)


def validate_treasury_df(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()

    # Required columns
    missing = [c for c in REQUIRED_CURVE_COLUMNS if c not in df.columns]
    if missing:
        report.errors.append(f"Missing required curve columns: {missing}")

    if "date" not in df.columns and df.index.name != "date":
        report.errors.append("No 'date' column or index found")

    present = [c for c in REQUIRED_CURVE_COLUMNS if c in df.columns]

    # Null fraction check
    for col in present:
        null_frac = df[col].isna().mean()
        if null_frac > MAX_NULL_FRACTION:
            report.warnings.append(
                f"Column {col!r} has {null_frac:.1%} null values (threshold {MAX_NULL_FRACTION:.0%})"
            )

    # Yield bounds
    for col in present:
        numeric = pd.to_numeric(df[col], errors="coerce")
        out_of_range = numeric.dropna()
        negatives = (out_of_range < YIELD_MIN).sum()
        too_high = (out_of_range > YIELD_MAX).sum()
        if negatives:
            report.warnings.append(f"Column {col!r}: {negatives} negative yield value(s)")
        if too_high:
            report.warnings.append(f"Column {col!r}: {too_high} yield(s) above {YIELD_MAX}%")

    # Large daily moves on 10Y
    if "10_yr" in df.columns:
        numeric_10y = pd.to_numeric(df["10_yr"], errors="coerce")
        daily_moves = numeric_10y.diff().abs() * 100  # bps
        big_moves = (daily_moves > MAX_DAILY_MOVE_BPS).sum()
        if big_moves:
            report.warnings.append(
                f"10_yr has {big_moves} daily move(s) exceeding {MAX_DAILY_MOVE_BPS} bps"
            )

    report.log_all()
    return report


def validate_feature_df(df: pd.DataFrame, fred_columns: list[str]) -> ValidationReport:
    report = ValidationReport()

    # All treasury cols must still be present
    missing_curve = [c for c in REQUIRED_CURVE_COLUMNS if c not in df.columns]
    if missing_curve:
        report.errors.append(f"Feature df missing curve columns: {missing_curve}")

    # FRED columns coverage
    for col in fred_columns:
        if col not in df.columns:
            report.warnings.append(f"FRED column {col!r} missing from feature df")
            continue
        null_frac = df[col].isna().mean()
        if null_frac > MAX_NULL_FRACTION:
            report.warnings.append(
                f"FRED column {col!r} has {null_frac:.1%} null values after forward-fill"
            )

    # Duplicate dates
    date_col = df.index if df.index.name == "date" else df.get("date")
    if date_col is not None:
        dupes = pd.Series(date_col).duplicated().sum()
        if dupes:
            report.errors.append(f"Feature df has {dupes} duplicate date(s)")

    report.log_all()
    return report
