from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

DEFAULT_TREASURY_DATASET_PATH = Path("data/daily_treasury_yield_curve.csv")
DEFAULT_OUTPUT_PATH = Path("data/model_features_daily.csv")
DEFAULT_START_DATE = "2023-01-01"

# Curated first-pass feature basket for 10Y modeling.
DEFAULT_FRED_SERIES: dict[str, str] = {
    "DFF": "fred_dff",
    "SOFR": "fred_sofr",
    "DGS3MO": "fred_dgs3mo",
    "DGS6MO": "fred_dgs6mo",
    "DGS1": "fred_dgs1",
    "DGS2": "fred_dgs2",
    "DGS10": "fred_dgs10",
    "DFII5": "fred_dfii5",
    "DFII10": "fred_dfii10",
    "DFII30": "fred_dfii30",
    "T10Y2Y": "fred_t10y2y",
    "T10YIE": "fred_t10yie",
    "T5YIE": "fred_t5yie",
    "T5YIFR": "fred_t5yifr",
    "VIXCLS": "fred_vix",
    "BAMLH0A0HYM2": "fred_hy_oas",
    "BAMLC0A0CM": "fred_ig_oas",
    "DEXUSEU": "fred_usdeur",
    "DCOILWTICO": "fred_wti",
}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "treasury-forecasting/1.0"})


@dataclass(frozen=True)
class FredBuildStats:
    rows_written: int
    output_path: Path
    fred_series_count: int
    min_date: str
    max_date: str


def _fetch_fred_series(
    series_id: str, api_key: str, start_date: str, retries: int = 3
) -> pd.DataFrame:
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&api_key={api_key}"
        "&file_type=json"
        f"&observation_start={start_date}"
    )
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep_s = 1.5 ** (attempt + 1)
                log.warning(
                    "FRED fetch attempt %d/%d failed for %s, retrying in %.1fs: %s",
                    attempt + 1, retries, series_id, sleep_s, exc,
                )
                time.sleep(sleep_s)
    else:
        raise RuntimeError(
            f"Failed to fetch FRED series {series_id} after {retries} attempts"
        ) from last_exc

    observations = payload.get("observations", [])
    rows: list[dict] = []
    for obs in observations:
        value = obs.get("value", ".")
        if value in {".", ""}:
            continue
        try:
            numeric = float(value)
        except ValueError:
            continue
        rows.append({"date": obs["date"], "value": numeric})
    return pd.DataFrame(rows)


def _fetch_one(
    series_id: str,
    out_col: str,
    api_key: str,
    series_start_date: str,
) -> tuple[str, str, pd.DataFrame | None]:
    """Worker for parallel FRED fetch; returns (series_id, out_col, df_or_None)."""
    try:
        df = _fetch_fred_series(series_id=series_id, api_key=api_key, start_date=series_start_date)
        log.debug("Fetched FRED series %s: %d observations", series_id, len(df))
        return series_id, out_col, df
    except RuntimeError as exc:
        log.warning("Skipping FRED series %s: %s", series_id, exc)
        return series_id, out_col, None


def build_fred_feature_dataset(
    treasury_dataset_path: Path,
    output_path: Path,
    api_key: str,
    start_date: str,
    full_refresh: bool = False,
    max_workers: int = 8,
) -> FredBuildStats:
    if not treasury_dataset_path.exists():
        raise FileNotFoundError(f"Treasury dataset not found: {treasury_dataset_path}")

    treasury = pd.read_csv(treasury_dataset_path)
    treasury["date"] = pd.to_datetime(treasury["date"])
    treasury = treasury.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")

    for col in treasury.columns:
        treasury[col] = pd.to_numeric(treasury[col], errors="coerce")

    existing_output: pd.DataFrame | None = None
    existing_fred_cols: list[str] = []
    effective_start_date = start_date

    if not full_refresh and output_path.exists():
        existing_output = pd.read_csv(output_path)
        if "date" in existing_output.columns:
            existing_output["date"] = pd.to_datetime(existing_output["date"])
            existing_output = (
                existing_output.sort_values("date")
                .drop_duplicates(subset=["date"])
                .set_index("date")
            )
            existing_fred_cols = [c for c in existing_output.columns if c.startswith("fred_")]
            if not existing_output.empty:
                latest_existing = existing_output.index.max().date()
                back_buffer = latest_existing - timedelta(days=10)
                effective_start_date = max(start_date, back_buffer.strftime("%Y-%m-%d"))

    merged = treasury.copy()
    if existing_output is not None and existing_fred_cols:
        merged = merged.join(existing_output[existing_fred_cols], how="left")

    # Determine per-series fetch params before parallelising.
    fetch_jobs: list[tuple[str, str, str]] = []
    for series_id, out_col in DEFAULT_FRED_SERIES.items():
        needs_full_history = full_refresh or out_col not in merged.columns
        if out_col in merged.columns:
            non_null = int(merged[out_col].notna().sum())
            if non_null == 0 or non_null < int(len(merged) * 0.9):
                needs_full_history = True
        series_start = start_date if needs_full_history else effective_start_date
        fetch_jobs.append((series_id, out_col, series_start))

    # Fetch all series in parallel.
    results: dict[str, tuple[str, pd.DataFrame | None]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, sid, col, api_key, sdate): (sid, col)
            for sid, col, sdate in fetch_jobs
        }
        for future in as_completed(futures):
            sid, out_col, df = future.result()
            results[sid] = (out_col, df)

    failed_series: list[str] = []
    for series_id, out_col, series_start in fetch_jobs:
        out_col_r, series = results.get(series_id, (out_col, None))
        needs_full_history = full_refresh or out_col not in merged.columns
        if out_col in merged.columns:
            non_null = int(merged[out_col].notna().sum())
            if non_null == 0 or non_null < int(len(merged) * 0.9):
                needs_full_history = True

        if series is None:
            failed_series.append(series_id)
            continue
        if series.empty:
            continue
        series["date"] = pd.to_datetime(series["date"])
        series = series.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
        series = series.rename(columns={"value": out_col})
        if out_col in merged.columns and needs_full_history:
            merged[out_col] = series[out_col].reindex(merged.index)
        elif out_col in merged.columns:
            aligned = series[out_col].reindex(merged.index)
            merged[out_col] = merged[out_col].where(~aligned.isna(), aligned)
        else:
            merged = merged.join(series[[out_col]], how="left")

    # Forward-fill slower-moving macro series on non-release days.
    fred_cols = [col for col in merged.columns if col.startswith("fred_")]
    if fred_cols:
        merged[fred_cols] = merged[fred_cols].ffill()

    merged = merged.reset_index()
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    if failed_series:
        log.warning("Skipped FRED series due to fetch errors: %s", ", ".join(failed_series))

    log.info(
        "FRED feature dataset complete: rows=%d series=%d date_range=[%s -> %s] path=%s",
        len(merged), len(fred_cols),
        str(merged["date"].min()), str(merged["date"].max()), output_path,
    )
    return FredBuildStats(
        rows_written=len(merged),
        output_path=output_path,
        fred_series_count=len(fred_cols),
        min_date=str(merged["date"].min()),
        max_date=str(merged["date"].max()),
    )


def _parse_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Date must be YYYY-MM-DD, got: {value!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m treasury_forecasting.fred_data",
        description="Build daily modeling feature dataset by merging Treasury curve with FRED series.",
    )
    parser.add_argument("--treasury-dataset-path", default=str(DEFAULT_TREASURY_DATASET_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, type=_parse_date,
                        help="YYYY-MM-DD")
    parser.add_argument("--api-key", default=os.getenv("FRED_API_KEY", ""))
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Fetch full FRED history from --start-date instead of incremental update.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel FRED fetch threads.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    if not args.api_key:
        raise SystemExit(
            "FRED API key required. Pass --api-key or set FRED_API_KEY environment variable."
        )

    stats = build_fred_feature_dataset(
        treasury_dataset_path=Path(args.treasury_dataset_path),
        output_path=Path(args.output_path),
        api_key=args.api_key,
        start_date=args.start_date,
        full_refresh=args.full_refresh,
        max_workers=args.max_workers,
    )
    print(
        "FRED feature dataset complete: "
        f"rows_written={stats.rows_written}, "
        f"fred_series_count={stats.fred_series_count}, "
        f"date_range=[{stats.min_date} -> {stats.max_date}], "
        f"output_path={stats.output_path}"
    )


if __name__ == "__main__":
    main()
