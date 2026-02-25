"""
Data ingestion and interactive column mapping for the Energy Model Regression Analyzer.
"""

import os
import pandas as pd
from typing import Dict, List, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """Load energy data from an Excel (.xlsx/.xls) or CSV file."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath)
    elif ext == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .xlsx or .csv.")
    return df


def detect_data_interval(df: pd.DataFrame, date_col: str) -> str:
    """
    Infer whether data is daily or monthly by examining the median gap between dates.
    Returns 'daily' or 'monthly'.
    """
    dates = pd.to_datetime(df[date_col]).dropna().sort_values().reset_index(drop=True)
    if len(dates) < 2:
        return 'monthly'
    diffs = dates.diff().dropna().dt.days
    median_gap = diffs.median()
    return 'daily' if median_gap <= 10 else 'monthly'


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def _display_columns(columns: List[str], df: pd.DataFrame) -> None:
    """Print a numbered list of columns with a sample value."""
    print()
    for i, col in enumerate(columns, 1):
        sample_vals = df[col].dropna()
        sample = str(sample_vals.iloc[0]) if len(sample_vals) > 0 else 'N/A'
        if len(sample) > 40:
            sample = sample[:37] + '...'
        print(f"  {i:3}. {col:<35s}  (e.g. {sample})")


def _pick_one(columns: List[str], df: pd.DataFrame, prompt: str) -> str:
    """Prompt the user to select exactly one column by number."""
    while True:
        _display_columns(columns, df)
        raw = input(f"\n{prompt}: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(columns):
                return columns[idx]
            print(f"  Please enter a number between 1 and {len(columns)}.")
        except ValueError:
            print("  Please enter a valid number.")


def _pick_many(columns: List[str], df: pd.DataFrame, prompt: str, optional: bool = False) -> List[str]:
    """Prompt the user to select one or more columns by comma-separated numbers."""
    while True:
        _display_columns(columns, df)
        if optional:
            print("  (Press Enter to skip)")
        raw = input(f"\n{prompt}: ").strip()
        if optional and raw == '':
            return []
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(',') if x.strip()]
            if not indices and not optional:
                print("  Please select at least one column.")
                continue
            selected = []
            valid = True
            for idx in indices:
                if 0 <= idx < len(columns):
                    selected.append(columns[idx])
                else:
                    print(f"  Invalid choice: {idx + 1}. Please re-enter.")
                    valid = False
                    break
            if valid:
                return selected
        except ValueError:
            print("  Please enter valid comma-separated numbers.")


def _identify_temp_col(iv_cols: List[str], df: pd.DataFrame, col_type: str) -> Optional[str]:
    """
    Auto-detect HDD or CDD column by name; if ambiguous, prompt the user.
    Returns the column name or None.
    """
    keyword = col_type.lower()
    matches = [c for c in iv_cols if keyword in c.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Multiple matches — ask user to pick
        print(f"\nMultiple columns may represent {col_type}. Which is correct?")
        for i, c in enumerate(matches, 1):
            print(f"  {i}. {c}")
        print(f"  0. None / Not applicable")
        raw = input("Enter number: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(matches):
                return matches[idx]
        except ValueError:
            pass
        return None
    # No auto-match — ask user
    print(f"\nWhich column represents {col_type} (Heating/Cooling Degree Days)?")
    for i, c in enumerate(iv_cols, 1):
        print(f"  {i}. {c}")
    print(f"  0. None / Not applicable")
    raw = input("Enter number: ").strip()
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(iv_cols):
            return iv_cols[idx]
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_columns(df: pd.DataFrame) -> Dict:
    """
    Interactively guide the user to map DataFrame columns to analysis roles.

    Returns a dict with keys:
        date_col, energy_col, iv_cols, mode_cols, hdd_col, cdd_col
    """
    all_cols = list(df.columns)

    print("\n" + "=" * 60)
    print("  COLUMN MAPPING")
    print("=" * 60)
    print(f"\n  {len(all_cols)} columns detected in the input file.")

    # --- Date column ---
    print("\nStep 1 of 4 — Select the DATE / PERIOD column:")
    date_col = _pick_one(all_cols, df, "Enter column number")
    remaining = [c for c in all_cols if c != date_col]

    # --- Energy (dependent) column ---
    print("\nStep 2 of 4 — Select the ENERGY CONSUMPTION column (dependent variable):")
    energy_col = _pick_one(remaining, df, "Enter column number")
    remaining = [c for c in remaining if c != energy_col]

    if not remaining:
        raise ValueError("No columns remain to use as independent variables.")

    # --- Independent variables ---
    print("\nStep 3 of 4 — Select INDEPENDENT VARIABLE column(s):")
    print("  (Comma-separated, e.g.  1,2,3  — select all that apply)")
    iv_cols = _pick_many(remaining, df, "Enter column number(s)")
    remaining = [c for c in remaining if c not in iv_cols]

    # --- Mode-of-operation flags ---
    mode_cols: List[str] = []
    if remaining:
        print("\nStep 4 of 4 — Select any MODE OF OPERATION flag columns")
        print("  (e.g. a weekday/weekend indicator column — optional):")
        mode_cols = _pick_many(remaining, df, "Enter column number(s)", optional=True)

    # --- Auto-identify HDD / CDD ---
    hdd_col = _identify_temp_col(iv_cols, df, 'HDD')
    cdd_col = _identify_temp_col(iv_cols, df, 'CDD')

    # --- Summary & confirmation ---
    mapping = {
        'date_col':   date_col,
        'energy_col': energy_col,
        'iv_cols':    iv_cols,
        'mode_cols':  mode_cols,
        'hdd_col':    hdd_col,
        'cdd_col':    cdd_col,
    }

    print("\n" + "-" * 60)
    print("  Mapping Summary")
    print("-" * 60)
    print(f"  Date column        : {date_col}")
    print(f"  Energy column      : {energy_col}")
    print(f"  Independent vars   : {', '.join(iv_cols)}")
    if mode_cols:
        print(f"  Mode flag cols     : {', '.join(mode_cols)}")
    if hdd_col:
        print(f"  HDD column         : {hdd_col}")
    if cdd_col:
        print(f"  CDD column         : {cdd_col}")
    print("-" * 60)

    confirm = input("\nConfirm this mapping? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\nRe-running column mapping...\n")
        return map_columns(df)

    return mapping
