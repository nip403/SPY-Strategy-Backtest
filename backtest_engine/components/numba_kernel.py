import numpy as np
import numba

# prefer contiguous array inputs (& non-broadcasts)

### Strategy ###

@numba.njit(cache=True)
def _resolve_positions(signals: np.ndarray) -> np.ndarray:
    """
    Sequentially resolve entry/exit/end-of-day signals into a held positions.
    Due to path-dependency this cannot be vectorised.

    Entry and exit conditions are price checks and not edge-triggered events.
    Exits are therefore state dependent (depending if a position exists) and take priority.

    signals : np.ndarray
        Boolean array, columns: long_entry, short_entry, long_exit, short_exit, end_of_day.

    Returns np.ndarray
        Resolved position (sign) per bar.
    """

    position = np.empty(len(signals))
    state = 0

    for i in range(len(signals)):
        le, se, lx, sx, eod = signals[i, 0], signals[i, 1], signals[i, 2], signals[i, 3], signals[i, 4]

        if eod:
            state = 0
        elif state > 0: # long
            if se: # short flip
                state = -1
            elif lx: # exit
                state = 0
        elif state < 0: # short
            if le: # long flip
                state = 1
            elif sx: # exit
                state = 0
        else: # entries
            if le:
                state = 1
            elif se:
                state = -1

        position[i] = state

    return position

### Execution ###

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

### Costing ###

@numba.njit(cache=True, parallel=True)
def _resolve_dynamic_impact(delta_leverage: np.ndarray, close: np.ndarray, volume: np.ndarray, adv: np.ndarray, ann_vol: np.ndarray, aum: np.ndarray, gross: np.ndarray, a1: float, a2: float, a3: float, a4: float, b1: float, commission: float) -> np.ndarray:
    """
    Numba core of DynamicCostModel.returns_matrix Kissel I-Star pricing.
    
    delta_leverage : np.ndarray
        abs(delta-position), shape (bars, aum_scenarios).
    close, volume, adv, ann_vol : np.ndarray
        Per-bar market/cost inputs, shape (bars,).
    aum : np.ndarray
        AUM scenarios, shape (aum_scenarios,).
    gross : np.ndarray
        Gross returns, shape (bars, aum_scenarios).
    a1, a2, a3, a4, b1, commission : float
        DynamicCostModel coefficients (see DynamicCostModel.returns_matrix docstring for the derivation).

    Returns np.ndarray
        Net returns, shape (bars, aum_scenarios).
    """

    n, k = delta_leverage.shape
    out = np.empty((n, k))

    for i in numba.prange(n):
        c = close[i]
        v = volume[i]
        a = adv[i]
        sigma = ann_vol[i]

        denom_adv = c * a
        denom_vol = c * v

        for j in range(k):
            dl = delta_leverage[i, j]

            adv_term = dl / denom_adv if denom_adv else 0
            participation = dl / denom_vol if denom_vol else 0

            perm = a1 * (adv_term ** a2) * (sigma ** a3) * dl
            temp = perm * (participation ** a4)
            comm = (commission * dl / c) if c else 0

            au = aum[j]
            mkt = (b1 * temp * (au ** (a2 + a4)) + (1 - b1) * perm * (au ** a2)) / 10_000

            out[i, j] = gross[i, j] - mkt - comm

    return out

### Decomposition ###

@numba.njit(cache=True, parallel=True)
def _resolve_split_long_short(position: np.ndarray, ret: np.ndarray, gross: np.ndarray, net: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba core of PortfolioDecomposer.split_long_short.
    Performs the long/short clip and gross/turnover/proportional cost split.
    Two passes turn out to be faster than one.
    
    position, ret, gross, net : np.ndarray
        (bars, scenarios) arrays. ret may be (bars, 1) to broadcast across scenarios.

    Returns tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (pos_long, pos_short, long_net_ret, short_net_ret), shapes (bars, scenarios).
    """

    n, k = position.shape
    pos_long = np.empty((n, k))
    pos_short = np.empty((n, k))

    for i in numba.prange(n):
        for j in range(k):
            p = position[i, j]
            pos_long[i, j] = p if p > 0 else 0
            pos_short[i, j] = p if p < 0 else 0

    long_ret = np.zeros((n, k))
    short_ret = np.zeros((n, k))
    ret_broadcast = ret.shape[1] == 1

    for i in numba.prange(1, n):
        for j in range(k):
            prev_long = pos_long[i - 1, j]
            prev_short = pos_short[i - 1, j]
            pl = pos_long[i, j]
            ps = pos_short[i, j]

            r = ret[i, 0] if ret_broadcast else ret[i, j]

            gross_long = prev_long * r
            gross_short = prev_short * r

            delta_long = abs(pl - prev_long)
            delta_short = abs(ps - prev_short)
            total_delta = delta_long + delta_short
            cost = gross[i, j] - net[i, j]

            if total_delta:
                cost_long = (delta_long / total_delta) * cost
                cost_short = (delta_short / total_delta) * cost
            else:
                cost_long = 0
                cost_short = 0

            long_ret[i, j] = gross_long - cost_long
            short_ret[i, j] = gross_short - cost_short

    return pos_long, pos_short, long_ret, short_ret

### LEGACY ###

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