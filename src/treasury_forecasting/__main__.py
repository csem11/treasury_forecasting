"""
Single entry point for the Treasury forecasting ETL pipeline.

Usage
-----
    python -m treasury_forecasting                  # full pipeline
    python -m treasury_forecasting --steps treasury # Treasury only
    python -m treasury_forecasting --steps fred     # FRED only
    python -m treasury_forecasting --full-refresh   # re-fetch all history

The .env file in the project root is loaded automatically — no need to
source it manually before running.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_dotenv(project_root: Path) -> None:
    """Parse .env and inject any missing keys into os.environ."""
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip().removeprefix("export").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m treasury_forecasting",
        description="Run the Treasury + FRED data ETL pipeline.",
    )
    parser.add_argument(
        "--steps",
        choices=["all", "treasury", "fred"],
        default="all",
        help="Which pipeline steps to run (default: all).",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Re-fetch all history instead of incremental update.",
    )
    parser.add_argument(
        "--backfill-start-year",
        type=int,
        default=2023,
        help="Earliest year to backfill when creating the Treasury dataset.",
    )
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Earliest date for FRED series (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for input/output CSV files.",
    )
    return parser


def _run_treasury(args: argparse.Namespace) -> bool:
    from treasury_forecasting.data_ingest import fetch_and_update_dataset

    data_dir = Path(args.data_dir)
    dataset_path = data_dir / "daily_treasury_yield_curve.csv"

    log.info("── Step 1 / Treasury yield curve ──────────────────────────")
    t0 = time.monotonic()
    try:
        stats = fetch_and_update_dataset(
            dataset_path=dataset_path,
            backfill_start_year=args.backfill_start_year,
            full_refresh=args.full_refresh,
        )
    except Exception as exc:
        log.error("Treasury fetch failed: %s", exc)
        return False

    elapsed = time.monotonic() - t0
    log.info(
        "   months fetched : %d", stats.months_requested
    )
    log.info("   rows fetched   : %d", stats.rows_fetched)
    log.info("   rows written   : %d  →  %s", stats.rows_written, stats.dataset_path)
    log.info("   elapsed        : %.1fs", elapsed)
    return True


def _run_fred(args: argparse.Namespace) -> bool:
    from treasury_forecasting.fred_data import build_fred_feature_dataset

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        log.error(
            "FRED_API_KEY not set. Add it to .env or pass via environment."
        )
        return False

    data_dir = Path(args.data_dir)
    treasury_path = data_dir / "daily_treasury_yield_curve.csv"
    output_path = data_dir / "model_features_daily.csv"

    if not treasury_path.exists():
        log.error(
            "Treasury dataset not found at %s — run Treasury step first.", treasury_path
        )
        return False

    log.info("── Step 2 / FRED macro features ────────────────────────────")
    t0 = time.monotonic()
    try:
        stats = build_fred_feature_dataset(
            treasury_dataset_path=treasury_path,
            output_path=output_path,
            api_key=api_key,
            start_date=args.start_date,
            full_refresh=args.full_refresh,
        )
    except Exception as exc:
        log.error("FRED fetch failed: %s", exc)
        return False

    elapsed = time.monotonic() - t0
    log.info("   series fetched : %d", stats.fred_series_count)
    log.info("   rows written   : %d  →  %s", stats.rows_written, stats.output_path)
    log.info("   date range     : %s  →  %s", stats.min_date, stats.max_date)
    log.info("   elapsed        : %.1fs", elapsed)
    return True


def main(argv: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parents[2]
    _load_dotenv(project_root)

    args = _build_parser().parse_args(argv)

    log.info("Treasury forecasting ETL  (steps=%s  full_refresh=%s)", args.steps, args.full_refresh)
    t_total = time.monotonic()
    ok = True

    if args.steps in ("all", "treasury"):
        ok = _run_treasury(args) and ok

    if args.steps in ("all", "fred"):
        ok = _run_fred(args) and ok

    elapsed_total = time.monotonic() - t_total
    if ok:
        log.info("── Pipeline complete  (%.1fs) ────────────────────────────", elapsed_total)
    else:
        log.error("── Pipeline finished with errors  (%.1fs) ───────────────", elapsed_total)
        sys.exit(1)


if __name__ == "__main__":
    main()
