"""Command-line entry point: fetch, forecast, verify and score."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quakecast import catalog, commit, etas, evaluate


def _utc(s: str) -> datetime:
    """Parse an ISO date or datetime as UTC."""
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download a regional ComCat catalog to parquet."""
    region = catalog.REGIONS[args.region]
    start, end = _utc(args.start), _utc(args.end)
    if args.full:
        df = catalog.fetch_fdsn(start, end, region, args.min_mag)
        default = f"data/{region.name}_comcat_full.parquet"
    else:
        df = catalog.fetch_comcat(start, end, region, args.min_mag)
        default = f"data/{region.name}_comcat.parquet"
    path = catalog.save(df, args.out or default)
    print(f"{len(df)} events -> {path}")


def cmd_forecast(args: argparse.Namespace) -> None:
    """Fit ETAS on the training window and commit a count and probability forecast."""
    df = catalog.load(args.catalog)
    df = df[df["mag"] >= args.m0]
    t0 = pd.Timestamp(_utc(args.start)) if args.start else df["time"].max().ceil("D")
    train_start = t0 - pd.Timedelta(days=365.25 * args.train_years)
    hist_start = train_start - pd.Timedelta(days=365.25 * args.burn_in_years)
    df = df[(df["time"] >= hist_start) & (df["time"] < t0)]
    times = catalog.to_days(df["time"], t0)
    mags = df["mag"].to_numpy(dtype=float)
    t_train = catalog.to_days(pd.Series([train_start]), t0)[0]
    params, ll = etas.fit(times, mags, t_train, 0.0, args.m0)
    b = etas.b_value(mags, args.m0)
    n_exp = etas.expected_count(params, times, mags, 0.0, float(args.days), args.m0)
    forecast = {
        "model": "temporal-ETAS-v0",
        "catalog": Path(args.catalog).name,
        "m0": args.m0,
        "window_start_utc": t0.isoformat(),
        "window_end_utc": (t0 + pd.Timedelta(days=args.days)).isoformat(),
        "expected_count": n_exp,
        "b_value": b,
        "prob_at_least_one": {
            f"M{m}": etas.prob_at_least_one(n_exp, args.m0, m, b) for m in args.targets
        },
        "params": params.to_dict(),
        "train_log_likelihood": ll,
        "branching_ratio": etas.branching_ratio(params, b),
    }
    path, digest = commit.commit_forecast(forecast, args.out_dir)
    print(json.dumps(forecast, indent=2))
    print(f"committed -> {path}\nsha256 {digest}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Confirm a commitment file still matches its hash."""
    ok = commit.verify(args.file)
    print("OK" if ok else "TAMPERED")
    raise SystemExit(0 if ok else 1)


def cmd_score(args: argparse.Namespace) -> None:
    """Score a committed forecast against a catalog that covers its window."""
    data = json.loads(Path(args.file).read_text())
    fc = data["record"]["forecast"]
    df = catalog.load(args.catalog)
    lo, hi = pd.Timestamp(fc["window_start_utc"]), pd.Timestamp(fc["window_end_utc"])
    if df["time"].max() < hi:
        raise SystemExit(f"catalog ends {df['time'].max()}, before window end {hi}")
    win = df[(df["time"] >= lo) & (df["time"] < hi)]
    n_obs = int((win["mag"] >= fc["m0"]).sum())
    probs, outcomes = [], []
    for key, p in fc["prob_at_least_one"].items():
        probs.append(p)
        outcomes.append(int((win["mag"] >= float(key[1:])).any()))
    result = {
        "verified": commit.verify(args.file),
        **evaluate.n_test(fc["expected_count"], n_obs),
        "poisson_log_score": evaluate.poisson_log_score(fc["expected_count"], n_obs),
        "brier": evaluate.brier_score(np.array(probs), np.array(outcomes)),
        "targets": dict(zip(fc["prob_at_least_one"], outcomes, strict=True)),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    """Parse arguments and dispatch to a subcommand."""
    ap = argparse.ArgumentParser(prog="quakecast")
    sub = ap.add_subparsers(required=True)

    f = sub.add_parser("fetch", help="download ComCat to parquet")
    f.add_argument("--region", default="california", choices=sorted(catalog.REGIONS))
    f.add_argument("--start", default="2000-01-01")
    f.add_argument("--end", default=datetime.now(UTC).date().isoformat())
    f.add_argument("--min-mag", type=float, default=2.5)
    f.add_argument("--out")
    f.add_argument(
        "--full", action="store_true", help="keep every ComCat column (ids, magType, quality)"
    )
    f.set_defaults(func=cmd_fetch)

    fc = sub.add_parser("forecast", help="fit ETAS and commit a forecast")
    fc.add_argument("catalog")
    fc.add_argument("--start", help="window start (default: day after last event)")
    fc.add_argument("--days", type=float, default=7.0)
    fc.add_argument("--m0", type=float, default=3.0)
    fc.add_argument("--train-years", type=float, default=5.0)
    fc.add_argument("--burn-in-years", type=float, default=1.0)
    fc.add_argument("--targets", type=float, nargs="+", default=[5.0, 6.0])
    fc.add_argument("--out-dir", default="commitments")
    fc.set_defaults(func=cmd_forecast)

    v = sub.add_parser("verify", help="check a commitment hash")
    v.add_argument("file")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("score", help="score a commitment once its window has closed")
    s.add_argument("file")
    s.add_argument("catalog")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
