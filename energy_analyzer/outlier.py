"""
Outlier detection for fitted regression models.

Uses Cook's Distance and internally studentized residuals.
Outliers are flagged for user review — never auto-excluded.
"""

from typing import Any, Dict, List

import numpy as np


def detect_outliers(
    model_result: Dict[str, Any],
    cooks_threshold: float = None,
    std_resid_threshold: float = 2.0,
) -> Dict[str, Any]:
    """
    Identify potential outliers in a fitted regression result.

    Cook's Distance threshold defaults to 4/n (a common rule of thumb).
    Standardized residual threshold defaults to 2 standard deviations.

    Parameters
    ----------
    model_result : dict returned by run_ols() or run_change_point().
    cooks_threshold : override for Cook's D cutoff (default 4/n).
    std_resid_threshold : |studentized residual| cutoff (default 2.0).

    Returns
    -------
    Dict with keys:
        cooks_flags      – row positions with Cook's D above threshold
        std_resid_flags  – row positions with |std resid| above threshold
        all_flags        – union of both sets (deduplicated, sorted)
        flag_count       – len(all_flags)
        flag_summary     – human-readable string listing up to 10 flagged positions
    """
    empty = {
        'cooks_flags':     [],
        'std_resid_flags': [],
        'all_flags':       [],
        'flag_count':      0,
        'flag_summary':    '',
    }

    fit = model_result.get('model_object')
    if fit is None:
        return empty

    n = int(fit.nobs)
    if n < 4:
        return empty

    if cooks_threshold is None:
        cooks_threshold = 4.0 / n

    try:
        influence = fit.get_influence()

        # Cook's Distance
        cooks_d, _ = influence.cooks_distance
        cooks_flags: List[int] = list(np.where(cooks_d > cooks_threshold)[0])

        # Internally studentized residuals
        std_resid = influence.resid_studentized_internal
        std_resid_flags: List[int] = list(
            np.where(np.abs(std_resid) > std_resid_threshold)[0]
        )

        all_flags = sorted(set(cooks_flags + std_resid_flags))
        flag_count = len(all_flags)
        flag_summary = (
            ', '.join(str(i) for i in all_flags[:10])
            + (' ...' if flag_count > 10 else '')
            if all_flags else ''
        )

        return {
            'cooks_flags':     cooks_flags,
            'std_resid_flags': std_resid_flags,
            'all_flags':       all_flags,
            'flag_count':      flag_count,
            'flag_summary':    flag_summary,
        }

    except Exception:
        return empty
