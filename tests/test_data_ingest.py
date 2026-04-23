from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from treasury_forecasting.data_ingest import (
    _fetch_month_rows,
    _merge_rows,
    _month_range,
    _normalize_header,
    fetch_and_update_dataset,
)


# ---- unit tests (no network) ----

def test_month_range_spans_year_boundary() -> None:
    assert _month_range(2023, 11, 2024, 2) == ["202311", "202312", "202401", "202402"]


def test_month_range_single_month() -> None:
    assert _month_range(2024, 3, 2024, 3) == ["202403"]


def test_merge_rows_prefers_new_values_and_sorts() -> None:
    existing = [
        {"date": "2024-01-03", "10_yr": "3.9"},
        {"date": "2024-01-02", "10_yr": "3.8"},
    ]
    fetched = [
        {"date": "2024-01-02", "10_yr": "3.85"},
        {"date": "2024-01-04", "10_yr": "3.95"},
    ]
    merged = _merge_rows(existing, fetched)
    assert [row["date"] for row in merged] == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert merged[0]["10_yr"] == "3.85"


def test_merge_rows_empty_existing() -> None:
    fetched = [{"date": "2024-01-02", "10_yr": "3.85"}]
    merged = _merge_rows([], fetched)
    assert len(merged) == 1


def test_normalize_header_known_maturities() -> None:
    assert _normalize_header("10 Yr") == "10_yr"
    assert _normalize_header("1.5 Month") == "1_5_mo"
    assert _normalize_header("Date") == "date"


def test_normalize_header_unknown_falls_back() -> None:
    assert _normalize_header("Some Col") == "some_col"


# ---- mocked network tests ----

SAMPLE_TREASURY_CSV = """Date,1 Mo,3 Mo,6 Mo,1 Yr,2 Yr,3 Yr,5 Yr,7 Yr,10 Yr,20 Yr,30 Yr
01/02/2024,5.60,5.42,5.23,5.00,4.50,4.35,4.20,4.22,4.10,4.40,4.30
01/03/2024,5.61,5.43,5.24,5.01,4.51,4.36,4.21,4.23,4.11,4.41,4.31
"""


def _mock_response(text: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


@patch("treasury_forecasting.data_ingest._SESSION")
def test_fetch_month_rows_parses_csv(mock_session: MagicMock) -> None:
    mock_session.get.return_value = _mock_response(SAMPLE_TREASURY_CSV)
    rows = _fetch_month_rows("202401")
    assert len(rows) == 2
    assert rows[0]["date"] == "2024-01-02"
    assert rows[0]["10_yr"] == "4.10"


@patch("treasury_forecasting.data_ingest._SESSION")
def test_fetch_month_rows_retries_on_failure(mock_session: MagicMock) -> None:
    import requests as req_lib

    error_resp = MagicMock()
    error_resp.raise_for_status.side_effect = req_lib.HTTPError("500")
    ok_resp = _mock_response(SAMPLE_TREASURY_CSV)
    mock_session.get.side_effect = [error_resp, ok_resp]

    with patch("treasury_forecasting.data_ingest.time.sleep"):
        rows = _fetch_month_rows("202401", retries=3)

    assert len(rows) == 2


@patch("treasury_forecasting.data_ingest._SESSION")
def test_fetch_and_update_dataset_writes_csv(mock_session: MagicMock, tmp_path: Path) -> None:
    mock_session.get.return_value = _mock_response(SAMPLE_TREASURY_CSV)
    out = tmp_path / "yields.csv"
    with patch("treasury_forecasting.data_ingest.date") as mock_date:
        mock_date.today.return_value = __import__("datetime").date(2024, 1, 31)
        stats = fetch_and_update_dataset(out, backfill_start_year=2024, full_refresh=True)
    assert out.exists()
    assert stats.rows_written >= 2
