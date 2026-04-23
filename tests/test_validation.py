import pandas as pd
import numpy as np
import pytest

from treasury_forecasting.validation import (
    REQUIRED_CURVE_COLUMNS,
    validate_feature_df,
    validate_treasury_df,
)


def _minimal_treasury_df() -> pd.DataFrame:
    n = 10
    # Use smooth incremental values so no daily move exceeds the 100 bps threshold.
    base = np.linspace(4.0, 4.1, n)
    data = {col: base + i * 0.01 for i, col in enumerate(REQUIRED_CURVE_COLUMNS)}
    data["date"] = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(data)


def test_valid_treasury_passes() -> None:
    df = _minimal_treasury_df()
    report = validate_treasury_df(df)
    assert report.ok
    assert len(report.warnings) == 0


def test_missing_curve_column_is_error() -> None:
    df = _minimal_treasury_df().drop(columns=["10_yr"])
    report = validate_treasury_df(df)
    assert not report.ok
    assert any("10_yr" in e for e in report.errors)


def test_negative_yield_is_warning() -> None:
    df = _minimal_treasury_df()
    df.loc[0, "10_yr"] = -0.5
    report = validate_treasury_df(df)
    assert any("negative" in w for w in report.warnings)


def test_high_null_fraction_is_warning() -> None:
    df = _minimal_treasury_df()
    df.loc[:, "30_yr"] = np.nan
    report = validate_treasury_df(df)
    assert any("30_yr" in w for w in report.warnings)


def test_feature_df_missing_fred_col_is_warning() -> None:
    df = _minimal_treasury_df().set_index("date")
    report = validate_feature_df(df, fred_columns=["fred_dff", "fred_vix"])
    assert any("fred_dff" in w for w in report.warnings)
