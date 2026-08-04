import numpy as np
import numba

@numba.njit(cache=True)
def _resolve_regime_starts(regime_target: np.ndarray, regime_maxcap: np.ndarray) -> np.ndarray:
    """
    Numba core of CappedVolumeRolloverExecution.fill_matrix's regime resolution.
    Greedily walks regime targets to derive each regime's starting position.

    regime_target : np.ndarray
        Target position per regime, shape (regimes,).
    regime_maxcap : np.ndarray
        Cumulative fill capacity reached by the end of each regime, one column per AUM, shape (regimes, aum_scenarios).

    Returns np.ndarray
        Starting position of each regime, one column per AUM, shape (regimes, aum_scenarios).
    """

    n_regimes, n_aum = regime_maxcap.shape
    p_start = np.empty((n_regimes, n_aum))
    p_current = np.zeros(n_aum)

    for i in range(n_regimes):
        t = regime_target[i]
        p_start[i] = p_current

        for j in range(n_aum):
            cur = p_current[j]
            cap = regime_maxcap[i, j]

            if t > cur:
                p_current[j] = min(t, cur + cap)
            elif t < cur:
                p_current[j] = max(t, cur - cap)

    return p_start

@numba.njit(cache=True)
def _resolve_ioc_fills(targets: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """
    Numba core of CappedVolumeExecution.fill_matrix's event resolution.
    Greedily walks signal-change events to derive each event's fill.
    
    targets : np.ndarray
        Target position at each signal-change event, shape (events,).
    caps : np.ndarray
        Per-event fill capacity, one column per AUM, shape (events, aum_scenarios).

    Returns np.ndarray
        Filled position at each event, one column per AUM, shape (events, aum_scenarios).
    """

    n_events, n_aum = caps.shape
    fills = np.empty((n_events, n_aum))
    p_current = np.zeros(n_aum)

    for i in range(n_events):
        t = targets[i]

        for j in range(n_aum):
            cur = p_current[j]
            cap = caps[i, j]

            if t > cur:
                p_current[j] = min(t, cur + cap)
            elif t < cur:
                p_current[j] = max(t, cur - cap)

            fills[i, j] = p_current[j]

    return fills

##### LEGACY #####

@numba.njit(cache=True)
def _resolve_window_fills(regime_starts: np.ndarray, window: np.ndarray, prev_target: np.ndarray, regime_target: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Numba core of RegimeVWAPExecution.fill's ramp resolution.
    Within each regime's window, ramps position from the target to target in proportion to cumulative volume and hold for remaining bars.

    regime_starts : np.ndarray
        Row index of each regime's first bar, shape (regimes,).
    window : np.ndarray
        Ramp length in bars for each regime, shape (regimes,). Guaranteed by the caller to be <= that
        regime's own length, so the ramp always completes within its own regime.
    prev_target : np.ndarray
        Position held immediately before each regime, shape (regimes,).
    regime_target : np.ndarray
        Target position of each regime, shape (regimes,).
    volume : np.ndarray
        Per-bar volume, shape (bars,).

    Returns np.ndarray
        Ramped position per bar, shape (bars,).
    """

    n = len(volume)
    position = np.empty(n)

    for i, start in enumerate(regime_starts):
        end = regime_starts[i + 1] if i + 1 < len(regime_starts) else n
        w = window[i]
        p0 = prev_target[i]
        t = regime_target[i]

        total = 0
        
        for j in range(start, start + w):
            total += volume[j]

        cum = 0
        for j in range(start, start + w):
            cum += volume[j]
            frac = cum / total if total > 0 else (j - start + 1) / w
            position[j] = p0 + frac * (t - p0)

        for j in range(start + w, end):
            position[j] = t

    return position