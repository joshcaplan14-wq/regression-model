#!/usr/bin/env python3
"""
Generate a realistic sample dataset for testing the Energy Model Regression Analyzer.

Produces three years of daily data for a hypothetical commercial building with:
  - Energy consumption (kWh) driven by HDD, CDD, and production
  - Realistic seasonal patterns and added noise
  - A handful of manually injected outliers

Usage
-----
    python generate_sample_data.py                  # saves sample_data.xlsx + sample_data.csv
    python generate_sample_data.py --format csv     # CSV only
"""

import argparse
import os

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate(output_dir: str = '.', fmt: str = 'both') -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # ---------------------------------------------------------------
    # Date range: 3 years of daily data
    # ---------------------------------------------------------------
    dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    n = len(dates)

    day_of_year = dates.day_of_year.to_numpy()

    # ---------------------------------------------------------------
    # Heating / Cooling Degree Days  (65°F base)
    # ---------------------------------------------------------------
    # Simulate daily average outdoor temperature with a seasonal sine wave
    # Peak ~85°F in summer, ~20°F in winter (northeast US climate)
    temp_mean = 52.5 + 32.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_daily = temp_mean + rng.normal(0, 8, size=n)

    base_temp = 65.0
    hdd = np.maximum(0.0, base_temp - temp_daily)
    cdd = np.maximum(0.0, temp_daily - base_temp)

    # ---------------------------------------------------------------
    # Production (arbitrary units — e.g. widgets/day)
    # ---------------------------------------------------------------
    # Higher on weekdays, zero on weekends, with seasonal ramp-up
    is_weekday = dates.dayofweek < 5
    production_base = np.where(is_weekday, 800, 0).astype(float)
    production_base += 50 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    production = np.maximum(0.0, production_base + rng.normal(0, 60, size=n))
    production[~is_weekday] = 0.0

    # ---------------------------------------------------------------
    # Energy consumption (kWh)
    # E = 1200 + 18*HDD + 22*CDD + 0.35*Production + noise
    # ---------------------------------------------------------------
    energy = (
        1200.0
        + 18.0 * hdd
        + 22.0 * cdd
        + 0.35 * production
        + rng.normal(0, 120, size=n)
    )

    # Inject ~10 outliers
    outlier_idx = rng.choice(n, size=10, replace=False)
    energy[outlier_idx] *= rng.uniform(0.5, 2.5, size=10)

    # Zero out energy on rare shutdowns
    shutdown_idx = rng.choice(n, size=5, replace=False)
    energy[shutdown_idx] = 0.0

    # ---------------------------------------------------------------
    # Build DataFrame
    # ---------------------------------------------------------------
    df = pd.DataFrame({
        'Date':       dates,
        'Energy_kWh': np.round(energy, 1),
        'HDD':        np.round(hdd, 1),
        'CDD':        np.round(cdd, 1),
        'Production': np.round(production, 0).astype(int),
        'Is_Weekday': is_weekday.astype(int),
    })

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    if fmt in ('xlsx', 'both'):
        xlsx_path = os.path.join(output_dir, 'sample_data.xlsx')
        df.to_excel(xlsx_path, index=False)
        print(f"Saved: {xlsx_path}")

    if fmt in ('csv', 'both'):
        csv_path = os.path.join(output_dir, 'sample_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    print(f"\nDataset: {len(df)} rows (daily, {dates[0].date()} – {dates[-1].date()})")
    print(f"Columns: {', '.join(df.columns)}")
    print("\nColumn mapping hints when running the analyzer:")
    print("  Date column       : Date")
    print("  Energy column     : Energy_kWh")
    print("  Independent vars  : HDD, CDD, Production")
    print("  Mode flag         : Is_Weekday  (optional)")
    print("  HDD column        : HDD")
    print("  CDD column        : CDD")


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate sample energy data for testing.')
    parser.add_argument('--output-dir', default='.', help='Directory to save output files')
    parser.add_argument(
        '--format', dest='fmt', choices=['xlsx', 'csv', 'both'], default='both',
        help='Output format (default: both)',
    )
    args = parser.parse_args()
    generate(output_dir=args.output_dir, fmt=args.fmt)


if __name__ == '__main__':
    main()
