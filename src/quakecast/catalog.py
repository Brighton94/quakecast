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


FDSN_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
FDSN_LIMIT = 20000

# Column meanings from the USGS ComCat CSV format documentation, after renaming in fetch_fdsn.
COLUMN_DESCRIPTIONS = {
    "time": "Origin time of the rupture (UTC)",
    "lat": "Epicentre latitude (decimal degrees)",
    "lon": "Epicentre longitude (decimal degrees)",
    "depth_km": "Hypocentre depth below the geoid (km); small negatives are above sea level",
    "mag": "Preferred magnitude",
    "mag_type": "Magnitude scale: ml local, mw moment, md duration, mh hand-picked, ...",
    "nst": "Number of seismic stations used to locate the event",
    "gap": "Largest azimuthal gap between stations (deg); >180 is a poorly constrained epicentre",
    "dmin": "Distance from epicentre to nearest station (degrees; 1 deg ~ 111 km)",
    "rms": "Root-mean-square travel-time residual of the location fit (s)",
    "net": "Network that contributed the preferred solution",
    "id": "Unique ComCat event id",
    "updated": "When the event was last revised (UTC)",
    "place": "Human-readable nearby place",
    "type": "Event type: earthquake, quarry blast, explosion, ...",
    "horizontal_error_km": "Horizontal location uncertainty (km)",
    "depth_error_km": "Depth uncertainty (km)",
    "mag_error": "Magnitude uncertainty",
    "mag_nst": "Number of stations used for the magnitude",
    "status": "automatic or reviewed by a seismologist",
    "location_source": "Network that produced the location",
    "mag_source": "Network that produced the magnitude",
}

_FDSN_RENAME = {
    "latitude": "lat",
    "longitude": "lon",
    "depth": "depth_km",
    "magType": "mag_type",
    "horizontalError": "horizontal_error_km",
    "depthError": "depth_error_km",
    "magError": "mag_error",
    "magNst": "mag_nst",
    "locationSource": "location_source",
    "magSource": "mag_source",
}


def fdsn_query_url(start: datetime, end: datetime, region: Region, min_mag: float) -> str:
    """Build a ComCat FDSN CSV query URL for one time slice."""
    params = {
        "format": "csv",
        "starttime": pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": pd.Timestamp(end).strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_mag,
        "minlatitude": region.min_lat,
        "maxlatitude": region.max_lat,
        "minlongitude": region.min_lon,
        "maxlongitude": region.max_lon,
        "orderby": "time-asc",
        "limit": FDSN_LIMIT,
    }
    return FDSN_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())


def tidy_fdsn(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename FDSN CSV columns, parse times and sort by origin time."""
    df = raw.rename(columns=_FDSN_RENAME)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["updated"] = pd.to_datetime(df["updated"], utc=True, format="ISO8601")
    df = df.drop_duplicates("id").sort_values("time").reset_index(drop=True)
    return df[[c for c in COLUMN_DESCRIPTIONS if c in df.columns]]


def fetch_fdsn(
    start: datetime,
    end: datetime,
    region: Region = CALIFORNIA,
    min_mag: float = 2.5,
    slice_days: float = 365.25,
) -> pd.DataFrame:
    """Download every ComCat column for a region, paging in time slices under the 20k limit.

    Unlike fetch_comcat this keeps event ids, magnitude type, location quality and event
    type, which the EDA and any later quality filtering need. A slice that comes back full
    is split in half and refetched.
    """
    frames = []
    step = pd.Timedelta(days=slice_days)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    stack = []
    while lo < hi:
        stack.append((lo, min(lo + step, hi)))
        lo += step
    stack.reverse()
    while stack:
        a, b = stack.pop()
        part = pd.read_csv(fdsn_query_url(a, b, region, min_mag))
        if len(part) >= FDSN_LIMIT:
            mid = a + (b - a) / 2
            stack += [(mid, b), (a, mid)]
            continue
        frames.append(part)
    return tidy_fdsn(pd.concat(frames, ignore_index=True))


def earthquakes_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop quarry blasts, explosions and other non-tectonic events when a type column exists."""
    if "type" not in df.columns:
        return df
    return df[df["type"] == "earthquake"].reset_index(drop=True)


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
