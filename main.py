#!/usr/bin/env python3
"""
Energy Model Regression Analyzer
=================================
IPMVP-aligned baseline regression modeling tool for energy management coaches.

Usage
-----
    python main.py <data_file> [options]

    python main.py data/site_daily.xlsx
    python main.py data/site_monthly.csv -o results/baseline_2024.xlsx

Options
-------
    -o, --output   Output file path  (default: energy_model_results.xlsx)
    -h, --help     Show this help message

Supported input formats : .xlsx, .xls, .csv
Supported output formats : .xlsx  (falls back to .csv if openpyxl is unavailable)
"""

import argparse
import os
import sys
import time

from energy_analyzer.ingest import load_data, map_columns
from energy_analyzer.models import run_all_models
from energy_analyzer.output import export_results, print_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='energy_analyzer',
        description=(
            'Energy Model Regression Analyzer — '
            'IPMVP-aligned baseline regression modeling'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'filepath',
        nargs='?',
        help='Path to input data file (.xlsx or .csv)',
    )
    parser.add_argument(
        '-o', '--output',
        default='energy_model_results.xlsx',
        metavar='OUTPUT',
        help='Output file path (default: energy_model_results.xlsx)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print()
    print("=" * 62)
    print("  Energy Model Regression Analyzer  |  v1.0")
    print("  IPMVP-aligned baseline modeling")
    print("=" * 62)

    # ------------------------------------------------------------------
    # 1. Resolve input file
    # ------------------------------------------------------------------
    filepath = args.filepath
    if not filepath:
        filepath = input("\nEnter path to input data file (.xlsx or .csv): ").strip()

    if not os.path.exists(filepath):
        print(f"\nError: File not found — {filepath}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data from: {filepath}")
    try:
        df = load_data(filepath)
    except Exception as exc:
        print(f"\nError loading file: {exc}")
        sys.exit(1)

    print(f"  {len(df):,} rows × {len(df.columns)} columns loaded successfully.")

    # ------------------------------------------------------------------
    # 3. Interactive column mapping
    # ------------------------------------------------------------------
    try:
        column_map = map_columns(df)
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sys.exit(0)
    except ValueError as exc:
        print(f"\nColumn mapping error: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Run all model permutations
    # ------------------------------------------------------------------
    print("\nRunning model permutations ...")
    t_start = time.time()

    try:
        results, best_model = run_all_models(df, column_map)
    except Exception as exc:
        print(f"\nError during model fitting: {exc}")
        raise

    elapsed = time.time() - t_start
    print(f"  Completed in {elapsed:.1f} seconds.")

    if not results:
        print("\nNo model results were produced. Check your data and column mapping.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Export results
    # ------------------------------------------------------------------
    output_path = args.output
    print(f"\nExporting results to: {output_path}")
    try:
        export_results(
            results,
            best_model,
            energy_col=column_map['energy_col'],
            output_path=output_path,
        )
    except Exception as exc:
        print(f"\nWarning: Export failed — {exc}")
        # Try CSV fallback
        csv_fallback = output_path.replace('.xlsx', '_fallback.csv')
        print(f"  Attempting CSV fallback: {csv_fallback}")
        try:
            import pandas as pd
            internal = {'_fitted', '_dates', '_actual'}
            rows = [{k: v for k, v in r.items() if k not in internal} for r in results]
            pd.DataFrame(rows).to_csv(csv_fallback, index=False)
            print(f"  Fallback saved: {csv_fallback}")
        except Exception:
            print("  Fallback also failed. Results not saved.")

    # ------------------------------------------------------------------
    # 6. Console summary
    # ------------------------------------------------------------------
    print_summary(results, best_model)


if __name__ == '__main__':
    main()
