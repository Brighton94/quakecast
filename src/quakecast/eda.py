"""Exploratory-analysis helpers for earthquake catalogs.

Three groups of tools:

* tabular checks that apply to any dataset (column overview, missing values, skewness,
  outliers),
* catalog checks from earthquake physics (magnitude of completeness, Gutenberg-Richter
  b-value, Omori-Utsu decay, declustering, inter-event times),
* time-series checks on daily counts, the quantity a next-day forecast predicts
  (stationarity, overdispersion, heteroskedasticity, seasonality, feature-target
  correlation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

EARTH_RADIUS_KM = 6371.0

# ---------------------------------------------------------------------------
# Tabular checks
# ---------------------------------------------------------------------------


def overview(df: pd.DataFrame, descriptions: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per column: dtype, non-null and missing counts, cardinality, an example value."""
    descriptions = descriptions or {}
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_pct": 100.0 * s.isna().mean() if len(s) else 0.0,
                "n_unique": int(s.nunique()),
                "example": non_null.iloc[0] if len(non_null) else None,
                "description": descriptions.get(col, ""),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and percentage per column, worst first, columns with none dropped."""
    out = pd.DataFrame({"missing": df.isna().sum(), "missing_pct": 100.0 * df.isna().mean()})
    out = out[out["missing"] > 0]
    return out.sort_values("missing", ascending=False)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """describe() for numeric columns plus median, skewness and excess kurtosis."""
    num = df.select_dtypes("number")
    out = num.describe().T
    out["median"] = num.median()
    out["skew"] = num.skew()
    out["kurtosis"] = num.kurt()
    return out


def iqr_bounds(x: np.ndarray, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences: values outside [Q1 - k*IQR, Q3 + k*IQR] are flagged."""
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def robust_z(x: np.ndarray) -> np.ndarray:
    """Modified z-score 0.6745 (x - median) / MAD, which heavy tails do not inflate."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def outlier_summary(
    df: pd.DataFrame, cols: list[str] | None = None, k: float = 1.5
) -> pd.DataFrame:
    """Per column: IQR fences, how many values fall outside, and how many have |robust z| > 3.5."""
    cols = cols or list(df.select_dtypes("number").columns)
    rows = []
    for col in cols:
        x = df[col].dropna().to_numpy(dtype=float)
        if len(x) == 0:
            continue
        lo, hi = iqr_bounds(x, k)
        n_iqr = int(((x < lo) | (x > hi)).sum())
        n_z = int((np.abs(robust_z(x)) > 3.5).sum())
        rows.append(
            {
                "column": col,
                "lower_fence": lo,
                "upper_fence": hi,
                "n_outside_iqr": n_iqr,
                "pct_outside_iqr": 100.0 * n_iqr / len(x),
                "n_robust_z_gt_3.5": n_z,
                "min": x.min(),
                "max": x.max(),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def value_counts_table(s: pd.Series, top: int = 10) -> pd.DataFrame:
    """Counts and shares of the most common values, with the rest pooled as 'other'."""
    vc = s.value_counts(dropna=False)
    head = vc.head(top)
    if len(vc) > top:
        head = pd.concat([head, pd.Series({"other": vc.iloc[top:].sum()})])
    return pd.DataFrame({"count": head, "pct": 100.0 * head / vc.sum()})


# ---------------------------------------------------------------------------
# Earthquake-physics checks
# ---------------------------------------------------------------------------


def bin_magnitudes(mags: np.ndarray, dm: float = 0.1) -> np.ndarray:
    """Round magnitudes to the catalog bin width so frequency counts are not split by noise."""
    return np.round(np.asarray(mags, dtype=float) / dm) * dm


def frequency_magnitude(mags: np.ndarray, dm: float = 0.1) -> pd.DataFrame:
    """Non-cumulative and cumulative (N >= M) counts per magnitude bin."""
    m = bin_magnitudes(mags, dm)
    bins = np.round(np.arange(m.min(), m.max() + dm / 2, dm), 6)
    idx = np.round((m - bins[0]) / dm).astype(int)
    n = np.bincount(idx, minlength=len(bins))
    return pd.DataFrame({"mag": bins, "count": n, "cum_count": n[::-1].cumsum()[::-1]})


def mc_maxc(mags: np.ndarray, dm: float = 0.1, correction: float = 0.2) -> float:
    """Magnitude of completeness by maximum curvature, plus the usual +0.2 correction.

    The modal bin of the non-cumulative distribution underestimates Mc, so Woessner and
    Wiemer (2005) recommend adding about 0.2.
    """
    fmd = frequency_magnitude(mags, dm)
    return float(np.round(fmd.loc[fmd["count"].idxmax(), "mag"] + correction, 6))


def b_value(mags: np.ndarray, mc: float, dm: float = 0.1) -> tuple[float, float]:
    """Aki-Utsu maximum-likelihood b-value with the Shi and Bolt (1982) standard error."""
    m = bin_magnitudes(mags, dm)
    m = m[m >= mc - 1e-9]
    n = len(m)
    if n < 2:
        return float("nan"), float("nan")
    mean = m.mean()
    b = np.log10(np.e) / (mean - (mc - dm / 2))
    se = 2.3 * b**2 * np.sqrt(np.sum((m - mean) ** 2) / (n * (n - 1)))
    return float(b), float(se)


def mc_over_time(
    df: pd.DataFrame, window: int = 1000, step: int = 250, dm: float = 0.1
) -> pd.DataFrame:
    """Mc by maximum curvature in sliding windows of events, timed at each window's centre."""
    t = df["time"].to_numpy()
    m = df["mag"].to_numpy(dtype=float)
    rows = []
    for start in range(0, max(len(df) - window, 0) + 1, step):
        sl = slice(start, start + window)
        rows.append({"time": t[start + window // 2], "mc": mc_maxc(m[sl], dm)})
    return pd.DataFrame(rows)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km; inputs broadcast."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def gardner_knopoff_window(mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gardner and Knopoff (1974) space (km) and time (days) windows for a mainshock."""
    mag = np.asarray(mag, dtype=float)
    dist = 10 ** (0.1238 * mag + 0.983)
    days = np.where(mag >= 6.5, 10 ** (0.032 * mag + 2.7389), 10 ** (0.5409 * mag - 0.547))
    return dist, days


def decluster_gk(df: pd.DataFrame) -> np.ndarray:
    """Window declustering: True for background (mainshock) events, False for clustered ones.

    Events are visited from largest to smallest. Each unflagged event flags every smaller
    event inside its space-time window, before or after it, as part of its cluster. This is
    crude next to ETAS-based stochastic declustering, but it is standard and good enough to
    ask whether the background rate is Poisson and free of seasonality.
    """
    t = ((df["time"] - df["time"].iloc[0]) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    mag = df["mag"].to_numpy(dtype=float)
    order = np.argsort(t, kind="stable")
    if not np.all(order == np.arange(len(t))):
        raise ValueError("catalog must be sorted by time")
    dist_w, time_w = gardner_knopoff_window(mag)
    background = np.ones(len(df), dtype=bool)
    for i in np.argsort(-mag, kind="stable"):
        if not background[i]:
            continue
        lo = np.searchsorted(t, t[i] - time_w[i], side="left")
        hi = np.searchsorted(t, t[i] + time_w[i], side="right")
        cand = np.arange(lo, hi)
        cand = cand[(cand != i) & background[cand] & (mag[cand] <= mag[i])]
        if len(cand) == 0:
            continue
        near = haversine_km(lat[i], lon[i], lat[cand], lon[cand]) <= dist_w[i]
        background[cand[near]] = False
    return background


def interevent_stats(times: pd.Series) -> dict:
    """Inter-event times in days and their coefficient of variation.

    A Poisson process has exponential gaps with CV = 1; clustering pushes CV above 1.
    """
    dt = np.diff(times.to_numpy()).astype("timedelta64[ns]").astype(float) / 86400e9
    dt = dt[dt > 0]
    return {
        "n": len(dt),
        "mean_days": float(dt.mean()),
        "median_days": float(np.median(dt)),
        "cv": float(dt.std() / dt.mean()),
    }


def omori_neg_loglik(theta: np.ndarray, dt: np.ndarray, t_end: float) -> float:
    """Negative log-likelihood of an Omori-Utsu rate K / (t + c)^p on [0, t_end]."""
    log_k, log_c, p = theta
    k, c = np.exp(log_k), np.exp(log_c)
    if abs(p - 1) < 1e-6:
        integral = k * (np.log(t_end + c) - np.log(c))
    else:
        integral = k * ((t_end + c) ** (1 - p) - c ** (1 - p)) / (1 - p)
    return float(integral - np.sum(np.log(k) - p * np.log(dt + c)))


def fit_omori(dt: np.ndarray, t_end: float) -> dict:
    """Maximum-likelihood Omori-Utsu fit to aftershock times (days after the mainshock)."""
    dt = np.asarray(dt, dtype=float)
    dt = dt[(dt > 0) & (dt <= t_end)]
    x0 = np.array([np.log(max(len(dt), 1) / 10), np.log(0.05), 1.1])
    res = minimize(
        omori_neg_loglik,
        x0,
        args=(dt, t_end),
        method="L-BFGS-B",
        bounds=[(-10, 15), (np.log(1e-5), np.log(5.0)), (0.3, 2.5)],
    )
    return {
        "K": float(np.exp(res.x[0])),
        "c": float(np.exp(res.x[1])),
        "p": float(res.x[2]),
        "n": len(dt),
        "converged": bool(res.success),
    }


def omori_rate(t: np.ndarray, k: float, c: float, p: float) -> np.ndarray:
    """Omori-Utsu aftershock rate per day."""
    return k / (np.asarray(t, dtype=float) + c) ** p


# ---------------------------------------------------------------------------
# Time-series checks on daily counts
# ---------------------------------------------------------------------------


def daily_counts(
    df: pd.DataFrame, m0: float, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> pd.Series:
    """Number of M >= m0 events per UTC day, with empty days filled by zero."""
    sel = df[df["mag"] >= m0]
    day = sel["time"].dt.floor("D")
    start = (start or df["time"].min()).floor("D")
    end = (end or df["time"].max()).floor("D")
    idx = pd.date_range(start, end, freq="D", tz=day.dt.tz)
    return day.value_counts().reindex(idx, fill_value=0).sort_index().rename("count")


def daily_features(df: pd.DataFrame, m0: float, m_target: float | None = None) -> pd.DataFrame:
    """Daily table of past-only features and the next-day target a forecast would predict.

    The target ``y_next`` is the number of M >= m_target events tomorrow. Every feature uses
    only today and earlier, so the table can be used for honest correlation checks and,
    later, as a baseline ML design matrix.
    """
    m_target = m0 if m_target is None else m_target
    sel = df[df["mag"] >= m0].copy()
    sel["day"] = sel["time"].dt.floor("D")
    idx = pd.date_range(
        df["time"].min().floor("D"), df["time"].max().floor("D"), freq="D", tz=sel["day"].dt.tz
    )
    g = sel.groupby("day")
    # Seismic moment in N m from Hanks-Kanamori, treating every magnitude as Mw.
    sel["moment"] = 10 ** (1.5 * sel["mag"] + 9.1)
    out = pd.DataFrame(index=idx)
    out["count"] = g.size().reindex(idx, fill_value=0)
    out["max_mag"] = g["mag"].max().reindex(idx).fillna(m0 - 0.1)
    out["mean_depth_km"] = g["depth_km"].mean().reindex(idx)
    out["log_moment"] = np.log10(
        sel.groupby("day")["moment"].sum().reindex(idx, fill_value=0) + 1.0
    )
    for w in (3, 7, 30, 365):
        out[f"count_{w}d"] = out["count"].rolling(w, min_periods=1).sum()
    out["max_mag_7d"] = out["max_mag"].rolling(7, min_periods=1).max()
    out["max_mag_30d"] = out["max_mag"].rolling(30, min_periods=1).max()
    big = (out["max_mag"] >= 5.0).to_numpy()
    since = np.full(len(out), np.nan)
    last = None
    for i, flag in enumerate(big):
        if flag:
            last = i
        if last is not None:
            since[i] = i - last
    out["days_since_m5"] = since
    out["log_days_since_m5"] = np.log10(out["days_since_m5"] + 1)
    out["day_of_week"] = out.index.day_of_week
    out["month"] = out.index.month
    tgt = df[df["mag"] >= m_target]["time"].dt.floor("D").value_counts()
    out["y_next"] = tgt.reindex(idx, fill_value=0).shift(-1)
    return out.iloc[:-1]


def feature_target_correlation(
    feats: pd.DataFrame, target: str = "y_next", exclude: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Pearson and Spearman correlation of every numeric feature with the target, with p-values.

    Spearman is the one to trust here: counts are heavy-tailed, so Pearson is driven by a
    handful of aftershock days.
    """
    rows = []
    y = feats[target]
    for col in feats.select_dtypes("number").columns:
        if col == target or col in exclude:
            continue
        ok = feats[col].notna() & y.notna()
        if ok.sum() < 3 or feats.loc[ok, col].nunique() < 2:
            continue
        r, pr = stats.pearsonr(feats.loc[ok, col], y[ok])
        rho, ps = stats.spearmanr(feats.loc[ok, col], y[ok])
        rows.append(
            {
                "feature": col,
                "pearson_r": r,
                "pearson_p": pr,
                "spearman_rho": rho,
                "spearman_p": ps,
                "n": int(ok.sum()),
            }
        )
    out = pd.DataFrame(rows).set_index("feature")
    return out.reindex(out["spearman_rho"].abs().sort_values(ascending=False).index)


def chunk_stats(x: pd.Series | np.ndarray, n_chunks: int = 10) -> pd.DataFrame:
    """Mean and variance of consecutive equal chunks: a quick look at stationarity."""
    arr = np.asarray(x, dtype=float)
    chunks = np.array_split(arr, n_chunks)
    return pd.DataFrame(
        {
            "mean": [c.mean() for c in chunks],
            "variance": [c.var(ddof=1) for c in chunks],
            "n": [len(c) for c in chunks],
        },
        index=pd.RangeIndex(1, n_chunks + 1, name="chunk"),
    )


def adf_test(x: pd.Series | np.ndarray, regression: str = "c") -> dict:
    """Augmented Dickey-Fuller test; a small p-value rejects a unit root (non-stationarity)."""
    import warnings

    from statsmodels.tsa.stattools import adfuller

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        stat, p, lags, nobs, crit, _ = adfuller(
            np.asarray(x, dtype=float), regression=regression, autolag="AIC"
        )
    return {
        "adf": float(stat),
        "p_value": float(p),
        "lags": int(lags),
        "nobs": int(nobs),
        **{f"crit_{k}": float(v) for k, v in crit.items()},
    }


def kpss_test(x: pd.Series | np.ndarray, regression: str = "c") -> dict:
    """KPSS test; the null is stationarity, so a small p-value means non-stationary.

    Run alongside ADF: the two have opposite nulls, and agreement is more convincing.
    """
    import warnings

    from statsmodels.tsa.stattools import kpss

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, lags, crit = kpss(np.asarray(x, dtype=float), regression=regression, nlags="auto")
    return {"kpss": float(stat), "p_value": float(p), "lags": int(lags)}


def dispersion_index(counts: pd.Series | np.ndarray) -> dict:
    """Variance-to-mean ratio of counts with a chi-square test against Poisson (ratio = 1)."""
    c = np.asarray(counts, dtype=float)
    n = len(c)
    mean, var = c.mean(), c.var(ddof=1)
    d = var / mean
    chi2 = (n - 1) * d
    p = float(stats.chi2.sf(chi2, n - 1))
    return {"mean": float(mean), "variance": float(var), "dispersion": float(d), "p_value": p}


def heteroskedasticity_tests(feats: pd.DataFrame, target: str, cols: list[str]) -> dict:
    """Breusch-Pagan and White tests on an OLS fit of the target on the given features.

    A small p-value means residual variance changes with the features: the noise is not
    constant, so least-squares error bars and MSE-trained models are misleading.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white

    data = feats[[target, *cols]].dropna()
    x = sm.add_constant(data[cols])
    res = sm.OLS(data[target], x).fit()
    bp = het_breuschpagan(res.resid, x)
    wh = het_white(res.resid, x)
    return {
        "breusch_pagan_lm": float(bp[0]),
        "breusch_pagan_p": float(bp[1]),
        "white_lm": float(wh[0]),
        "white_p": float(wh[1]),
        "r2": float(res.rsquared),
    }


def arch_test(x: pd.Series | np.ndarray, nlags: int = 7) -> dict:
    """Engle's ARCH LM test: does today's variance depend on recent squared deviations?"""
    import warnings

    from statsmodels.stats.diagnostic import het_arch

    arr = np.asarray(x, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        lm, lm_p, f, f_p = het_arch(arr - arr.mean(), nlags=nlags)
    return {"lm": float(lm), "p_value": float(lm_p)}


def seasonal_profile(times: pd.Series, by: str) -> pd.DataFrame:
    """Event counts by hour, day of week or month, with a chi-square test for uniformity.

    For months the expectation is weighted by days per month so February is not flagged.
    """
    t = pd.DatetimeIndex(times)
    if by == "hour":
        keys, levels, weights = t.hour, np.arange(24), np.ones(24)
    elif by == "dayofweek":
        keys, levels, weights = t.day_of_week, np.arange(7), np.ones(7)
    elif by == "month":
        keys, levels = t.month, np.arange(1, 13)
        weights = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    else:
        raise ValueError("by must be 'hour', 'dayofweek' or 'month'")
    obs = pd.Series(keys).value_counts().reindex(levels, fill_value=0).to_numpy()
    exp = obs.sum() * weights / weights.sum()
    chi2, p = stats.chisquare(obs, exp)
    out = pd.DataFrame({by: levels, "observed": obs, "expected": exp, "ratio": obs / exp})
    out.attrs.update({"chi2": float(chi2), "p_value": float(p)})
    return out.set_index(by)


def stl_monthly(counts: pd.Series, period: int = 12):
    """STL decomposition of monthly counts into trend, seasonal and residual parts."""
    from statsmodels.tsa.seasonal import STL

    monthly = counts.resample("MS").sum()
    return STL(monthly.astype(float), period=period, robust=True).fit()


def seasonal_share(res) -> float:
    """Share of non-trend variance in an STL fit that the seasonal component explains."""
    seas, resid = np.asarray(res.seasonal), np.asarray(res.resid)
    return float(np.var(seas) / (np.var(seas) + np.var(resid)))


def seasonal_share_null(counts: pd.Series, n_sims: int = 50, seed: int = 0) -> np.ndarray:
    """Seasonal share STL finds in Poisson noise with the same daily mean and length.

    STL always extracts some 'seasonal' pattern, even from noise, so an observed share only
    means something next to this baseline.
    """
    rng = np.random.default_rng(seed)
    lam = float(np.mean(counts))
    return np.array(
        [
            seasonal_share(stl_monthly(pd.Series(rng.poisson(lam, len(counts)), counts.index)))
            for _ in range(n_sims)
        ]
    )


# ---------------------------------------------------------------------------
# One-call summary
# ---------------------------------------------------------------------------


def summary_report(df: pd.DataFrame, m0: float = 3.0) -> dict:
    """Headline EDA numbers for a catalog, as plain JSON-ready values.

    Non-earthquake events are dropped first when an event-type column is present.
    """
    eq = df[df["type"] == "earthquake"] if "type" in df.columns else df
    eq = eq.reset_index(drop=True)
    mags = eq["mag"].to_numpy(dtype=float)
    mc = mc_maxc(mags)
    b, b_se = b_value(mags, max(mc, m0))
    above = eq[eq["mag"] >= m0].reset_index(drop=True)
    bg = decluster_gk(above)
    counts = daily_counts(eq, m0)
    bg_counts = daily_counts(above[bg], m0, counts.index[0], counts.index[-1])
    feats = daily_features(eq, m0)
    corr = feature_target_correlation(feats, exclude=("day_of_week", "month"))
    return {
        "n_events": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_non_earthquake": int(len(df) - len(eq)),
        "start": str(eq["time"].min()),
        "end": str(eq["time"].max()),
        "mc_maxc": mc,
        "b_value": b,
        "b_value_se": b_se,
        "m0": m0,
        "n_above_m0": int(len(above)),
        "background_fraction": float(bg.mean()),
        "interevent_cv_all": interevent_stats(above["time"])["cv"],
        "interevent_cv_background": interevent_stats(above["time"][bg])["cv"],
        "daily_dispersion_all": dispersion_index(counts)["dispersion"],
        "daily_dispersion_background": dispersion_index(bg_counts)["dispersion"],
        "adf_p": adf_test(counts)["p_value"],
        "arch_p": arch_test(counts)["p_value"],
        "month_seasonality_p_all": seasonal_profile(above["time"], "month").attrs["p_value"],
        "month_seasonality_p_background": seasonal_profile(above["time"][bg], "month").attrs[
            "p_value"
        ],
        "top_features_spearman": corr["spearman_rho"].head(5).round(3).to_dict(),
    }
