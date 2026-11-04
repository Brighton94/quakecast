"""Checks for the EDA helpers on small synthetic catalogs with known answers."""

import numpy as np
import pandas as pd
import pytest

from quakecast import catalog, eda


def _gr_mags(n: int, b: float, mc: float, seed: int = 0, dm: float = 0.1) -> np.ndarray:
    """Gutenberg-Richter magnitudes above mc, binned like a real catalog."""
    rng = np.random.default_rng(seed)
    m = (mc - dm / 2) + rng.exponential(1 / (b * np.log(10)), n)
    return eda.bin_magnitudes(m, dm)


def _poisson_catalog(n_days: int, rate: float, seed: int = 0, box: float = 1.0) -> pd.DataFrame:
    """Uniform-in-time events in a box of side `box` degrees, M >= 3."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate * n_days)
    t0 = pd.Timestamp("2000-01-01", tz="UTC")
    days = np.sort(rng.uniform(0, n_days, n))
    return pd.DataFrame(
        {
            "time": t0 + pd.to_timedelta(days, unit="D"),
            "lat": rng.uniform(34.5 - box / 2, 34.5 + box / 2, n),
            "lon": rng.uniform(-117.5 - box / 2, -117.5 + box / 2, n),
            "depth_km": rng.uniform(0, 15, n),
            "mag": _gr_mags(n, 1.0, 3.0, seed),
        }
    )


# Tabular -------------------------------------------------------------------


def test_overview_and_missing():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "y"]})
    ov = eda.overview(df, {"a": "first"})
    assert ov.loc["a", "missing"] == 1
    assert ov.loc["b", "n_unique"] == 2
    assert ov.loc["a", "description"] == "first"
    ms = eda.missing_summary(df)
    assert list(ms.index) == ["a"]
    assert ms.loc["a", "missing_pct"] == pytest.approx(100 / 3)


def test_numeric_summary_skew_sign():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"right": rng.exponential(size=5000), "sym": rng.normal(size=5000)})
    s = eda.numeric_summary(df)
    assert s.loc["right", "skew"] > 1.5
    assert abs(s.loc["sym", "skew"]) < 0.15
    assert {"median", "kurtosis"} <= set(s.columns)


def test_outliers_flag_planted_values():
    x = np.r_[np.random.default_rng(2).normal(size=1000), 50.0, -40.0]
    out = eda.outlier_summary(pd.DataFrame({"x": x}))
    assert out.loc["x", "n_robust_z_gt_3.5"] >= 2
    lo, hi = eda.iqr_bounds(x)
    assert lo < -2 and hi > 2


def test_robust_z_constant_input():
    assert np.all(eda.robust_z(np.ones(5)) == 0)


def test_value_counts_pools_tail():
    vc = eda.value_counts_table(pd.Series(list("aaabbc")), top=2)
    assert vc.loc["other", "count"] == 1
    assert vc["pct"].sum() == pytest.approx(100)


# Earthquake physics --------------------------------------------------------


def test_b_value_recovers_truth():
    b, se = eda.b_value(_gr_mags(20000, 1.0, 2.5), 2.5)
    assert b == pytest.approx(1.0, abs=0.03)
    assert 0 < se < 0.03


def test_mc_maxc_finds_completeness():
    # Detection probability is a normal CDF centred on M3 (Ogata and Katsura 1993).
    from scipy.stats import norm

    full = _gr_mags(50000, 1.0, 2.0, seed=3)
    keep = np.random.default_rng(4).uniform(size=full.size) < norm.cdf(full, 3.0, 0.1)
    mc = eda.mc_maxc(full[keep])
    assert 2.9 <= mc <= 3.4


def test_frequency_magnitude_cumulative():
    fmd = eda.frequency_magnitude(np.array([3.0, 3.0, 3.1, 3.4]))
    assert fmd["cum_count"].iloc[0] == 4
    assert fmd["count"].sum() == 4
    assert np.all(np.diff(fmd["cum_count"]) <= 0)


def test_haversine_one_degree_latitude():
    assert eda.haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.1)


def test_gk_windows_grow_with_magnitude():
    d, t = eda.gardner_knopoff_window(np.array([3.0, 5.0, 7.0]))
    assert np.all(np.diff(d) > 0) and np.all(np.diff(t) > 0)


def test_decluster_removes_planted_aftershocks():
    # A wide box keeps most background events outside the M6 window (~53 km, ~500 days).
    bg = _poisson_catalog(3650, 0.05, seed=5, box=10.0)
    t0 = pd.Timestamp("2005-06-01", tz="UTC")
    main = pd.DataFrame(
        {"time": [t0], "lat": [34.5], "lon": [-117.5], "depth_km": [8.0], "mag": [6.0]}
    )
    rng = np.random.default_rng(6)
    n_after = 200
    after = pd.DataFrame(
        {
            "time": t0 + pd.to_timedelta(np.sort(rng.exponential(5, n_after)) + 1e-3, unit="D"),
            "lat": 34.5 + rng.normal(0, 0.05, n_after),
            "lon": -117.5 + rng.normal(0, 0.05, n_after),
            "depth_km": 8.0,
            "mag": np.full(n_after, 3.0),
        }
    )
    df = pd.concat([bg, main, after]).sort_values("time").reset_index(drop=True)
    keep = eda.decluster_gk(df)
    is_after = df["time"].isin(after["time"]) & (df["mag"] == 3.0)
    assert keep[is_after.to_numpy()].mean() < 0.05
    assert keep[(df["mag"] == 6.0).to_numpy()].all()
    assert keep.sum() > 0.8 * len(bg)


def test_decluster_requires_sorted_times():
    df = _poisson_catalog(100, 1.0).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        eda.decluster_gk(df)


def test_interevent_cv_near_one_for_poisson():
    s = eda.interevent_stats(_poisson_catalog(5000, 2.0, seed=7)["time"])
    assert s["cv"] == pytest.approx(1.0, abs=0.05)


def test_fit_omori_recovers_p():
    rng = np.random.default_rng(8)
    k, c, p, t_end = 100.0, 0.05, 1.1, 365.0
    # Inverse-CDF sampling of the Omori-Utsu density on [0, t_end].
    total = (c ** (1 - p) - (t_end + c) ** (1 - p)) / (p - 1)
    n = rng.poisson(k * total)
    u = rng.uniform(size=n) * total
    t = (c ** (1 - p) - u * (p - 1)) ** (1 / (1 - p)) - c
    fit = eda.fit_omori(t, t_end)
    assert fit["converged"]
    assert fit["p"] == pytest.approx(p, abs=0.1)
    assert eda.omori_rate(np.array([0.0]), 1.0, 1.0, 1.0)[0] == pytest.approx(1.0)


# Time series ---------------------------------------------------------------


def test_daily_counts_fills_empty_days():
    t = pd.to_datetime(["2020-01-01 03:00", "2020-01-01 05:00", "2020-01-04 01:00"], utc=True)
    df = pd.DataFrame({"time": t, "mag": [3.0, 2.0, 3.5]})
    c = eda.daily_counts(df, 2.5)
    assert c.tolist() == [1, 0, 0, 1]


def test_daily_features_target_is_tomorrow_and_no_leak():
    df = _poisson_catalog(400, 1.0, seed=9)
    f = eda.daily_features(df, 3.0)
    counts = eda.daily_counts(df, 3.0)
    assert f["y_next"].iloc[:-1].tolist() == counts.iloc[1:-1].tolist()
    assert (f["count_7d"] >= f["count"]).all()
    # Features at day d must not change if tomorrow's events are removed.
    cut = df["time"].min().floor("D") + pd.Timedelta(days=200)
    f_cut = eda.daily_features(df[df["time"] < cut + pd.Timedelta(days=1)], 3.0)
    cols = [c for c in f.columns if c != "y_next"]
    pd.testing.assert_frame_equal(f_cut[cols].iloc[:200], f[cols].iloc[:200], check_freq=False)


def test_feature_target_correlation_finds_signal():
    rng = np.random.default_rng(10)
    x = rng.normal(size=500)
    f = pd.DataFrame(
        {"x": x, "noise": rng.normal(size=500), "y_next": 2 * x + rng.normal(size=500)}
    )
    corr = eda.feature_target_correlation(f)
    assert corr.index[0] == "x"
    assert corr.loc["x", "spearman_p"] < 1e-10
    assert corr.loc["noise", "spearman_p"] > 1e-3


def test_chunk_stats_shapes():
    cs = eda.chunk_stats(np.arange(100), 10)
    assert len(cs) == 10 and cs["n"].sum() == 100
    assert np.all(np.diff(cs["mean"]) > 0)


def test_adf_and_kpss_agree_on_white_noise_and_random_walk():
    rng = np.random.default_rng(11)
    noise = rng.normal(size=1000)
    walk = np.cumsum(noise)
    assert eda.adf_test(noise)["p_value"] < 0.01
    assert eda.adf_test(walk)["p_value"] > 0.05
    assert eda.kpss_test(noise)["p_value"] > 0.05
    assert eda.kpss_test(walk)["p_value"] < 0.05


def test_dispersion_poisson_vs_clustered():
    rng = np.random.default_rng(12)
    assert eda.dispersion_index(rng.poisson(3, 5000))["p_value"] > 0.01
    clustered = rng.negative_binomial(1, 0.25, 5000)
    d = eda.dispersion_index(clustered)
    assert d["dispersion"] > 2 and d["p_value"] < 1e-6


def test_heteroskedasticity_detects_variance_growing_with_x():
    rng = np.random.default_rng(13)
    x = rng.uniform(0, 10, 2000)
    f = pd.DataFrame({"x": x, "y": x + rng.normal(size=2000) * x})
    het = eda.heteroskedasticity_tests(f, "y", ["x"])
    assert het["breusch_pagan_p"] < 1e-6
    f["y"] = x + rng.normal(size=2000)
    assert eda.heteroskedasticity_tests(f, "y", ["x"])["breusch_pagan_p"] > 1e-3


def test_arch_detects_volatility_clustering():
    rng = np.random.default_rng(14)
    calm, wild = rng.normal(0, 1, 500), rng.normal(0, 8, 500)
    x = np.concatenate([calm, wild, calm, wild])
    assert eda.arch_test(x)["p_value"] < 1e-6
    assert eda.arch_test(rng.normal(size=2000))["p_value"] > 1e-3


def test_seasonal_profile_detects_daytime_blasts():
    rng = np.random.default_rng(15)
    base = pd.Timestamp("2010-01-01", tz="UTC")
    uniform = base + pd.to_timedelta(rng.uniform(0, 3650, 5000), unit="D")
    prof = eda.seasonal_profile(pd.Series(uniform), "hour")
    assert prof.attrs["p_value"] > 1e-3
    day = pd.to_timedelta(rng.integers(0, 3650, 2000), unit="D")
    blasts = base + day + pd.to_timedelta(rng.uniform(18, 23, 2000), unit="h")
    prof2 = eda.seasonal_profile(pd.Series(uniform.append(blasts)), "hour")
    assert prof2.attrs["p_value"] < 1e-10
    assert prof2["ratio"].idxmax() in range(18, 24)


def test_seasonal_profile_month_weights_and_bad_key():
    rng = np.random.default_rng(16)
    t = pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(rng.uniform(0, 36525, 20000), "D")
    assert eda.seasonal_profile(pd.Series(t), "month").attrs["p_value"] > 1e-3
    with pytest.raises(ValueError):
        eda.seasonal_profile(pd.Series(t), "week")


def test_stl_recovers_annual_cycle():
    idx = pd.date_range("2000-01-01", periods=365 * 20, freq="D", tz="UTC")
    rate = 5 + 3 * np.sin(2 * np.pi * idx.day_of_year / 365.25)
    counts = pd.Series(np.random.default_rng(17).poisson(rate), index=idx)
    res = eda.stl_monthly(counts)
    assert np.std(res.seasonal) > 3 * np.std(res.resid)


def test_seasonal_share_matches_null_for_noise():
    idx = pd.date_range("2000-01-01", periods=365 * 20, freq="D", tz="UTC")
    noise = pd.Series(np.random.default_rng(20).poisson(1.0, len(idx)), index=idx)
    null = eda.seasonal_share_null(noise, n_sims=10)
    share = eda.seasonal_share(eda.stl_monthly(noise))
    assert null.min() - 0.1 < share < null.max() + 0.1
    rate = 1 + 0.8 * np.sin(2 * np.pi * idx.day_of_year / 365.25)
    seasonal = pd.Series(np.random.default_rng(21).poisson(rate), index=idx)
    assert eda.seasonal_share(eda.stl_monthly(seasonal)) > null.max()


# Catalog -------------------------------------------------------------------


def test_tidy_fdsn_renames_sorts_and_dedups():
    raw = pd.DataFrame(
        {
            "time": [
                "2020-01-02T00:00:00.000Z",
                "2020-01-01T00:00:00.000Z",
                "2020-01-01T00:00:00.000Z",
            ],
            "latitude": [34.0, 35.0, 35.0],
            "longitude": [-117.0, -118.0, -118.0],
            "depth": [5.0, 7.0, 7.0],
            "mag": [3.1, 3.4, 3.4],
            "magType": ["ml", "mw", "mw"],
            "id": ["b", "a", "a"],
            "updated": ["2020-02-01T00:00:00.000Z"] * 3,
            "type": ["earthquake", "quarry blast", "quarry blast"],
        }
    )
    df = catalog.tidy_fdsn(raw)
    assert list(df["id"]) == ["a", "b"]
    assert {"lat", "lon", "depth_km", "mag_type"} <= set(df.columns)
    assert str(df["time"].dt.tz) == "UTC"
    assert set(df.columns) <= set(catalog.COLUMN_DESCRIPTIONS)
    assert list(catalog.earthquakes_only(df)["id"]) == ["b"]


def test_fdsn_url_has_limit_and_box():
    url = catalog.fdsn_query_url(
        pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), catalog.CALIFORNIA, 2.5
    )
    assert "limit=20000" in url and "minmagnitude=2.5" in url and "format=csv" in url


# Summary and plots ---------------------------------------------------------


def test_summary_report_on_poisson_catalog():
    df = _poisson_catalog(2000, 1.0, seed=18, box=10.0)
    df["type"] = "earthquake"
    df.loc[df.index[:5], "type"] = "quarry blast"
    rep = eda.summary_report(df, 3.0)
    assert rep["n_non_earthquake"] == 5
    assert rep["b_value"] == pytest.approx(1.0, abs=0.1)
    assert rep["daily_dispersion_all"] == pytest.approx(1.0, abs=0.15)
    assert rep["interevent_cv_background"] == pytest.approx(1.0, abs=0.1)
    assert len(rep["top_features_spearman"]) == 5


def test_plots_render(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from quakecast import plots

    plots.set_style()
    df = _poisson_catalog(1000, 1.0, seed=19)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    plots.run_sequence_plot(eda.daily_counts(df, 3.0), "counts", "n", ax=ax[0])
    fmd = eda.frequency_magnitude(df["mag"].to_numpy())
    plots.fmd_plot(fmd, 3.0, 1.0, ax=ax[1])
    plots.seasonal_bars({"all": eda.seasonal_profile(df["time"], "dayofweek")}, "dayofweek", ax[2])
    fig.savefig(tmp_path / "x.png")
    assert (tmp_path / "x.png").stat().st_size > 1000
