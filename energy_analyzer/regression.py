"""
Regression engine for OLS (linear / multivariable) and change-point (3-parameter) models.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Minimum observations required to attempt a fit
MIN_OBS = 6


def run_ols(X: pd.DataFrame, y: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Fit an OLS model:  y = b0 + b1*x1 + b2*x2 + ...

    Parameters
    ----------
    X : DataFrame of independent variable(s), rows already cleaned of NaN.
    y : Series of the dependent variable, same index as X.

    Returns
    -------
    Dict with model statistics, or None on failure.
    """
    if len(y) < MIN_OBS:
        return None

    try:
        X_const = sm.add_constant(X, has_constant='add')
        fit = sm.OLS(y, X_const).fit()

        iv_names = list(X.columns)
        n_iv = len(iv_names)

        result: Dict[str, Any] = {
            'model_type':    'Linear' if n_iv == 1 else 'Multivariable',
            'intercept':     float(fit.params.get('const', fit.params.iloc[0])),
            'coefficients':  {v: float(fit.params[v]) for v in iv_names if v in fit.params},
            'r_squared':     float(fit.rsquared),
            'p_values':      {v: float(fit.pvalues[v]) for v in iv_names if v in fit.pvalues},
            't_stats':       {v: float(fit.tvalues[v]) for v in iv_names if v in fit.tvalues},
            'n_obs':         int(fit.nobs),
            'fitted_values': fit.fittedvalues.copy(),
            'residuals':     fit.resid.copy(),
            'model_object':  fit,
            'changepoint':   None,
        }
        return result

    except Exception:
        return None


def run_change_point(
    x: pd.Series,
    y: pd.Series,
    n_grid: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    Fit a 3-parameter change-point model:
        E = b0 + b1 * max(0, x - changepoint)

    The changepoint is found by grid search over the 5th–95th percentile of x.

    Parameters
    ----------
    x : Series for the single independent variable (HDD or CDD), index-aligned with y.
    y : Series for the dependent variable.
    n_grid : Number of candidate changepoints to test.

    Returns
    -------
    Dict with model statistics (including 'changepoint'), or None on failure.
    """
    if len(y) < MIN_OBS:
        return None

    x_arr = x.values.astype(float)
    y_series = y.copy()

    cp_lo = np.percentile(x_arr, 5)
    cp_hi = np.percentile(x_arr, 95)

    if cp_lo >= cp_hi:
        return None

    cp_grid = np.linspace(cp_lo, cp_hi, n_grid)

    best_r2: float = -np.inf
    best_result: Optional[Dict[str, Any]] = None
    best_cp: Optional[float] = None

    iv_name = x.name

    for cp in cp_grid:
        x_transformed = np.maximum(0.0, x_arr - cp)

        # Skip if all zero (no data above changepoint)
        if x_transformed.sum() == 0:
            continue

        X_cp = pd.DataFrame({'_cp_feat': x_transformed}, index=x.index)
        candidate = run_ols(X_cp, y_series)

        if candidate is not None and candidate['r_squared'] > best_r2:
            best_r2 = candidate['r_squared']
            best_result = candidate
            best_cp = cp

    if best_result is None:
        return None

    # Rename the internal '_cp_feat' key back to the original variable name
    best_result['model_type'] = 'Change-Point'
    best_result['changepoint'] = float(best_cp)

    if '_cp_feat' in best_result['coefficients']:
        best_result['coefficients'] = {iv_name: best_result['coefficients']['_cp_feat']}
    if '_cp_feat' in best_result['p_values']:
        best_result['p_values'] = {iv_name: best_result['p_values']['_cp_feat']}
    if '_cp_feat' in best_result['t_stats']:
        best_result['t_stats'] = {iv_name: best_result['t_stats']['_cp_feat']}

    return best_result
