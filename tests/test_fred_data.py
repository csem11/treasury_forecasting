import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from treasury_forecasting.fred_data import (
    DEFAULT_FRED_SERIES,
    _fetch_fred_series,
    build_fred_feature_dataset,
    build_parser,
)


# ---- unit tests ----

def test_build_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.treasury_dataset_path.endswith("data/daily_treasury_yield_curve.csv")
    assert args.output_path.endswith("data/model_features_daily.csv")
    assert args.start_date == "2023-01-01"


def test_build_fred_feature_dataset_requires_treasury_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_fred_feature_dataset(
            treasury_dataset_path=tmp_path / "missing.csv",
            output_path=tmp_path / "out.csv",
            api_key="dummy",
            start_date="2023-01-01",
        )


def test_default_fred_series_has_policy_and_real_rate_proxies() -> None:
    assert "DFF" in DEFAULT_FRED_SERIES
    assert "SOFR" in DEFAULT_FRED_SERIES
    assert "DFII5" in DEFAULT_FRED_SERIES
    assert "DFII30" in DEFAULT_FRED_SERIES


def test_parse_date_rejects_bad_format() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--start-date", "2023/01/01"])


# ---- mocked network tests ----

def _fred_payload(series_id: str, values: list[tuple[str, str]]) -> str:
    obs = [{"date": d, "value": v} for d, v in values]
    return json.dumps({"observations": obs})


def _mock_response(body: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json.loads(body)
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


@patch("treasury_forecasting.fred_data._SESSION")
def test_fetch_fred_series_parses_observations(mock_session: MagicMock) -> None:
    payload = _fred_payload("DFF", [("2024-01-02", "5.33"), ("2024-01-03", "5.33"), ("2024-01-04", ".")])
    mock_session.get.return_value = _mock_response(payload)
    df = _fetch_fred_series("DFF", api_key="dummy", start_date="2024-01-01")
    # "." values are dropped
    assert len(df) == 2
    assert list(df.columns) == ["date", "value"]
    assert df["value"].iloc[0] == pytest.approx(5.33)


@patch("treasury_forecasting.fred_data._SESSION")
def test_fetch_fred_series_retries_then_raises(mock_session: MagicMock) -> None:
    import requests as req_lib

    err = MagicMock()
    err.raise_for_status.side_effect = req_lib.HTTPError("500")
    mock_session.get.return_value = err

    with patch("treasury_forecasting.fred_data.time.sleep"):
        with pytest.raises(RuntimeError, match="Failed to fetch FRED series"):
            _fetch_fred_series("DFF", api_key="dummy", start_date="2024-01-01", retries=2)

    assert mock_session.get.call_count == 2


MINIMAL_TREASURY_CSV = """date,3_mo,2_yr,10_yr,30_yr
2024-01-02,5.42,4.50,4.10,4.30
2024-01-03,5.43,4.51,4.11,4.31
"""

FRED_SERIES_COUNT = len(DEFAULT_FRED_SERIES)


@patch("treasury_forecasting.fred_data._fetch_one")
def test_build_fred_feature_dataset_merges_series(
    mock_fetch_one: MagicMock, tmp_path: Path
) -> None:
    treasury_path = tmp_path / "treasury.csv"
    treasury_path.write_text(MINIMAL_TREASURY_CSV)
    output_path = tmp_path / "out.csv"

    # Return a simple 2-row series for every FRED series.
    def _fake_fetch(series_id: str, out_col: str, api_key: str, series_start_date: str):
        df = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "value": [1.0, 1.1]})
        df = df.rename(columns={"value": out_col})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return series_id, out_col, df.reset_index().rename(columns={"index": "date"})

    mock_fetch_one.side_effect = _fake_fetch

    stats = build_fred_feature_dataset(
        treasury_dataset_path=treasury_path,
        output_path=output_path,
        api_key="dummy",
        start_date="2024-01-01",
        max_workers=1,
    )
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert "date" in df.columns
    assert stats.rows_written == 2
