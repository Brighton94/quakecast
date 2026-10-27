"""Fetch the USGS ComCat catalog through pyCSEP and store it as a tidy table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Region:
    """Lat/lon bounding box for a catalog query."""

    name: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


CALIFORNIA = Region("california", 31.5, 43.0, -125.4, -113.1)
ITALY = Region("italy", 36.0, 48.0, 6.0, 19.0)
JAPAN = Region("japan", 24.0, 46.0, 122.0, 146.0)
NEW_ZEALAND = Region("new_zealand", -48.0, -34.0, 165.0, 179.0)
REGIONS = {r.name: r for r in (CALIFORNIA, ITALY, JAPAN, NEW_ZEALAND)}


def fetch_comcat(
    start: datetime, end: datetime, region: Region = CALIFORNIA, min_mag: float = 2.5
) -> pd.DataFrame:
    """Download ComCat events in a region and time window as a DataFrame."""
    import csep

    cat = csep.query_comcat(
        start_time=start,
        end_time=end,
        min_magnitude=min_mag,
        min_latitude=region.min_lat,
        max_latitude=region.max_lat,
        min_longitude=region.min_lon,
        max_longitude=region.max_lon,
        verbose=False,
    )
    return catalog_to_frame(cat)


def catalog_to_frame(cat) -> pd.DataFrame:
    """Convert a pyCSEP catalog into a time-sorted DataFrame."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(np.asarray(cat.get_epoch_times()), unit="ms", utc=True),
            "lat": cat.get_latitudes(),
            "lon": cat.get_longitudes(),
            "depth_km": cat.get_depths(),
            "mag": cat.get_magnitudes(),
        }
    )
    return df.sort_values("time").reset_index(drop=True)


def save(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a catalog DataFrame to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load(path: str | Path) -> pd.DataFrame:
    """Read a catalog DataFrame from parquet."""
    return pd.read_parquet(path)


def to_days(times: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    """Convert timestamps to float days since an origin."""
    return ((times - origin) / pd.Timedelta(days=1)).to_numpy(dtype=float)
