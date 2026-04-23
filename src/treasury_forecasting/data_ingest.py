from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)

TREASURY_CSV_BASE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/all/{month}"
)
DEFAULT_DATASET_PATH = Path("data/daily_treasury_yield_curve.csv")
DATE_FMT_SOURCE = "%m/%d/%Y"
DATE_FMT_DATASET = "%Y-%m-%d"

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "treasury-forecasting/1.0"})


@dataclass(frozen=True)
class FetchStats:
    months_requested: int
    rows_fetched: int
    rows_written: int
    dataset_path: Path


def _month_range(start_year: int, start_month: int, end_year: int, end_month: int) -> list[str]:
    month_keys: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        month_keys.append(f"{year}{month:02d}")
        month += 1
        if month > 12:
            year += 1
            month = 1
    return month_keys


def _normalize_header(name: str) -> str:
    normalized = name.strip().lower()
    replacements = {
        "date": "date",
        "1 mo": "1_mo",
        "1.5 month": "1_5_mo",
        "2 mo": "2_mo",
        "3 mo": "3_mo",
        "4 mo": "4_mo",
        "6 mo": "6_mo",
        "1 yr": "1_yr",
        "2 yr": "2_yr",
        "3 yr": "3_yr",
        "5 yr": "5_yr",
        "7 yr": "7_yr",
        "10 yr": "10_yr",
        "20 yr": "20_yr",
        "30 yr": "30_yr",
    }
    return replacements.get(normalized, normalized.replace(" ", "_").replace(".", "_"))


def _build_month_csv_url(month_key: str) -> str:
    query = urlencode(
        {
            "type": "daily_treasury_yield_curve",
            "field_tdr_date_value_month": month_key,
            "page": "",
            "_format": "csv",
        }
    )
    return f"{TREASURY_CSV_BASE_URL.format(month=month_key)}?{query}"


def _fetch_month_rows(month_key: str, retries: int = 3) -> list[dict[str, str]]:
    url = _build_month_csv_url(month_key)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=30)
            resp.raise_for_status()
            content = resp.text
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep_s = 1.5 ** (attempt + 1)
                log.warning("Treasury fetch attempt %d/%d failed for %s, retrying in %.1fs: %s",
                            attempt + 1, retries, month_key, sleep_s, exc)
                time.sleep(sleep_s)
    else:
        raise RuntimeError(
            f"Failed to fetch month {month_key} from Treasury endpoint after {retries} attempts"
        ) from last_exc

    rows: list[dict[str, str]] = []
    reader = csv.DictReader(content.splitlines())
    for row in reader:
        if not row:
            continue
        parsed_date = datetime.strptime(row["Date"].strip(), DATE_FMT_SOURCE).date()
        normalized: dict[str, str] = {"date": parsed_date.strftime(DATE_FMT_DATASET)}
        for source_key, value in row.items():
            if source_key == "Date":
                continue
            normalized[_normalize_header(source_key)] = (value or "").strip()
        rows.append(normalized)
    return rows


def _read_existing_rows(dataset_path: Path) -> list[dict[str, str]]:
    if not dataset_path.exists():
        return []
    with dataset_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def _resolve_start_period(existing_rows: list[dict[str, str]], backfill_start_year: int) -> tuple[int, int]:
    if not existing_rows:
        return backfill_start_year, 1
    max_date = max(datetime.strptime(row["date"], DATE_FMT_DATASET).date() for row in existing_rows)
    return max_date.year, max_date.month


def _merge_rows(existing_rows: list[dict[str, str]], fetched_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_date: dict[str, dict[str, str]] = {row["date"]: row for row in existing_rows if row.get("date")}
    for row in fetched_rows:
        by_date[row["date"]] = row
    return [by_date[key] for key in sorted(by_date.keys())]


def _write_rows(dataset_path: Path, rows: list[dict[str, str]]) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    with dataset_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def fetch_and_update_dataset(
    dataset_path: Path, backfill_start_year: int, full_refresh: bool = False
) -> FetchStats:
    existing_rows = [] if full_refresh else _read_existing_rows(dataset_path)
    start_year, start_month = _resolve_start_period(existing_rows, backfill_start_year)
    today = date.today()
    months = _month_range(start_year, start_month, today.year, today.month)

    fetched: list[dict[str, str]] = []
    for month_key in months:
        log.debug("Fetching Treasury month %s", month_key)
        fetched.extend(_fetch_month_rows(month_key))

    merged_rows = _merge_rows(existing_rows, fetched)
    _write_rows(dataset_path, merged_rows)

    log.info(
        "Treasury ingest complete: months=%d rows_fetched=%d rows_written=%d path=%s",
        len(months), len(fetched), len(merged_rows), dataset_path,
    )
    return FetchStats(
        months_requested=len(months),
        rows_fetched=len(fetched),
        rows_written=len(merged_rows),
        dataset_path=dataset_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="treasury_forecasting",
        description="Fetch and maintain Treasury daily yield curve data.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help="Local CSV dataset path to update.",
    )
    parser.add_argument(
        "--backfill-start-year",
        type=int,
        default=2023,
        help="Earliest year to backfill when creating a dataset.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Rebuild dataset from backfill start year instead of incremental append.",
    )
    return parser


def run_ingest_from_args(argv: list[str] | None = None) -> FetchStats:
    parser = build_parser()
    args = parser.parse_args(argv)
    return fetch_and_update_dataset(
        dataset_path=Path(args.dataset_path),
        backfill_start_year=args.backfill_start_year,
        full_refresh=args.full_refresh,
    )
