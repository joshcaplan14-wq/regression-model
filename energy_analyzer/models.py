"""
Model permutation generation and orchestration for the Energy Model Regression Analyzer.

Dimensions tested per the spec:
  - Data interval      : daily, monthly
  - Baseline period    : 1 yr, 2 yr, 3 yr
  - Independent vars   : all non-empty subsets of the user-supplied IV columns
  - Model type         : Linear / Multivariable OLS; Change-Point (HDD/CDD only)
  - Operating mode     : All data / Weekday / Weekend (weekday/weekend = daily only)
"""

import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .outlier import detect_outliers
from .regression import run_change_point, run_ols

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# Statistical pass/fail thresholds (spec §3.3)
# -------------------------------------------------------------------
P_VALUE_THRESHOLD = 0.05
T_STAT_THRESHOLD = 2.0
R_SQUARED_THRESHOLD = 0.75

# Minimum rows required for a model to be fitted
MIN_ROWS = 6


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _get_baseline_slice(df: pd.DataFrame, date_col: str, years: int) -> pd.DataFrame:
    """Return rows within the most-recent `years` years of data."""
    dates = pd.to_datetime(df[date_col])
    max_date = dates.max()
    cutoff = max_date - pd.DateOffset(years=years)
    return df[dates >= cutoff].copy()


def _aggregate_to_monthly(
    df: pd.DataFrame,
    date_col: str,
    energy_col: str,
    iv_cols: List[str],
) -> pd.DataFrame:
    """
    Sum daily rows into calendar months.
    All energy and IV columns are summed (appropriate for degree-days and production totals).
    """
    df = df.copy()
    df['_period'] = pd.to_datetime(df[date_col]).dt.to_period('M')
    agg_cols = {energy_col: 'sum', **{c: 'sum' for c in iv_cols}}
    monthly = df.groupby('_period').agg(agg_cols).reset_index()
    monthly[date_col] = monthly['_period'].dt.to_timestamp()
    return monthly.drop(columns=['_period'])


def _variable_subsets(iv_cols: List[str]) -> List[List[str]]:
    """All non-empty subsets of iv_cols, ordered by size then name."""
    subsets = []
    for r in range(1, len(iv_cols) + 1):
        for combo in combinations(iv_cols, r):
            subsets.append(list(combo))
    return subsets


def _passes_thresholds(result: Dict[str, Any]) -> bool:
    """Return True if the model clears all three IPMVP thresholds."""
    if result['r_squared'] < R_SQUARED_THRESHOLD:
        return False
    for p in result['p_values'].values():
        if p >= P_VALUE_THRESHOLD:
            return False
    for t in result['t_stats'].values():
        if abs(t) <= T_STAT_THRESHOLD:
            return False
    return True


# -------------------------------------------------------------------
# Single-model runner
# -------------------------------------------------------------------

def _run_one_model(
    df: pd.DataFrame,
    date_col: str,
    energy_col: str,
    iv_cols: List[str],
    model_type: str,
    cp_var: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Fit one regression model on a pre-filtered DataFrame.

    Parameters
    ----------
    df         : Rows for a specific interval / baseline / mode.
    date_col   : Name of the date column (used only to capture chart data).
    energy_col : Dependent variable column.
    iv_cols    : Independent variable columns for this permutation.
    model_type : 'Linear', 'Multivariable', or 'Change-Point'.
    cp_var     : Column name for the changepoint variable (Change-Point only).

    Returns
    -------
    dict with regression output + chart data, or None.
    """
    relevant_cols = [date_col, energy_col] + iv_cols
    df_clean = df[relevant_cols].dropna(subset=[energy_col] + iv_cols)
    n_excluded = len(df) - len(df_clean)

    if len(df_clean) < MIN_ROWS:
        return None

    y = df_clean[energy_col]
    dates = pd.to_datetime(df_clean[date_col])

    if model_type == 'Change-Point':
        if cp_var is None:
            return None
        result = run_change_point(df_clean[cp_var], y)
    else:
        X = df_clean[iv_cols]
        result = run_ols(X, y)

    if result is None:
        return None

    result['excluded_rows'] = n_excluded
    result['n_data'] = len(df_clean)

    # Outlier detection
    outlier_info = detect_outliers(result)
    result['outlier_flag_count'] = outlier_info['flag_count']
    result['outlier_flag_summary'] = outlier_info['flag_summary']

    # Store chart data (aligned with fitted_values index)
    result['chart_dates'] = dates.values
    result['chart_actual'] = y.values

    return result


# -------------------------------------------------------------------
# Main orchestrator
# -------------------------------------------------------------------

def run_all_models(
    df: pd.DataFrame,
    column_map: Dict[str, Any],
) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Test all valid model permutations and return (results_list, best_model_row).

    Each element of results_list is a flat dict ready for DataFrame export.
    best_model_row is the passing model with the highest R², or None.
    """
    date_col   = column_map['date_col']
    energy_col = column_map['energy_col']
    iv_cols    = column_map['iv_cols']
    hdd_col    = column_map.get('hdd_col')
    cdd_col    = column_map.get('cdd_col')

    # Columns eligible for change-point modelling (temperature-driven)
    cp_eligible = {c for c in [hdd_col, cdd_col] if c}

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Detect native interval
    from .ingest import detect_data_interval
    native_interval = detect_data_interval(df, date_col)
    print(f"  Detected native data interval : {native_interval}")

    # Intervals to test
    intervals = ['daily', 'monthly'] if native_interval == 'daily' else ['monthly']

    var_subsets = _variable_subsets(iv_cols)

    results: List[Dict] = []
    model_counter = 0
    tested_count = 0

    for interval in intervals:
        # Build interval-level DataFrame
        if interval == 'monthly' and native_interval == 'daily':
            df_interval = _aggregate_to_monthly(df, date_col, energy_col, iv_cols)
        else:
            df_interval = df.copy()

        for baseline_years in [1, 2, 3]:
            df_bl = _get_baseline_slice(df_interval, date_col, baseline_years)
            if len(df_bl) < MIN_ROWS:
                continue

            # Operating modes
            if interval == 'daily':
                df_bl = df_bl.copy()
                df_bl['_dow'] = pd.to_datetime(df_bl[date_col]).dt.dayofweek
                modes = [
                    ('All',     df_bl),
                    ('Weekday', df_bl[df_bl['_dow'] < 5]),
                    ('Weekend', df_bl[df_bl['_dow'] >= 5]),
                ]
            else:
                # Monthly — no weekday/weekend split (spec §6)
                modes = [('All', df_bl)]

            for mode_label, df_mode in modes:
                # Drop helper column if present
                df_mode = df_mode.drop(columns=['_dow'], errors='ignore').copy()
                if len(df_mode) < MIN_ROWS:
                    continue

                for var_combo in var_subsets:
                    # Verify all selected vars are present (always true unless aggregation dropped one)
                    if not all(v in df_mode.columns for v in var_combo):
                        continue

                    # Determine which model types apply to this variable combination
                    model_specs: List[Tuple[str, Optional[str]]] = []

                    if len(var_combo) == 1:
                        model_specs.append(('Linear', None))
                        # Change-point only for a single HDD or CDD variable
                        if var_combo[0] in cp_eligible:
                            model_specs.append(('Change-Point', var_combo[0]))
                    else:
                        model_specs.append(('Multivariable', None))

                    for mtype, cp_var in model_specs:
                        model_counter += 1
                        tested_count += 1

                        result = _run_one_model(
                            df_mode, date_col, energy_col, var_combo, mtype, cp_var
                        )

                        if result is None:
                            continue

                        passed = _passes_thresholds(result)

                        row: Dict[str, Any] = {
                            'Model ID':                    f'M{model_counter:05d}',
                            'Interval':                    interval.capitalize(),
                            'Baseline Period':             f'{baseline_years}yr',
                            'Independent Variables':       ', '.join(var_combo),
                            'Model Type':                  result['model_type'],
                            'Operating Mode Segment':      mode_label,
                            'Intercept':                   round(result['intercept'], 4),
                            'Change-Point Value':          (
                                round(result['changepoint'], 4)
                                if result.get('changepoint') is not None else ''
                            ),
                            'R²':                          round(result['r_squared'], 4),
                            'Pass / Fail':                 'Pass' if passed else 'Fail',
                            'Outlier Flags':               result['outlier_flag_count'],
                            'Outlier Flag Positions':      result['outlier_flag_summary'],
                            'Excluded Rows (missing data)': result['excluded_rows'],
                            'N Observations':              result['n_data'],
                            'Recommended':                 'No',
                        }

                        # Per-variable stats — one column per IV
                        for var in iv_cols:
                            row[f'Coeff_{var}']   = round(result['coefficients'].get(var, np.nan), 6)
                            row[f'p-value_{var}'] = round(result['p_values'].get(var, np.nan),     6)
                            row[f't-stat_{var}']  = round(result['t_stats'].get(var, np.nan),      4)

                        # Internal keys for chart generation (excluded from CSV/Excel)
                        row['_fitted'] = result.get('fitted_values')
                        row['_dates']  = result.get('chart_dates')
                        row['_actual'] = result.get('chart_actual')

                        results.append(row)

                        if tested_count % 100 == 0:
                            print(f"  ...{tested_count} models fitted so far")

    print(f"\n  Total model permutations tested : {tested_count}")

    # Identify recommended model
    passing = [r for r in results if r['Pass / Fail'] == 'Pass']
    best_model: Optional[Dict] = None

    if passing:
        best_model = max(passing, key=lambda r: r['R²'])
        best_model['Recommended'] = 'Yes'
        print(f"  Models passing all thresholds   : {len(passing)}")
    else:
        print("  WARNING: No models passed all statistical thresholds.")

    return results, best_model
