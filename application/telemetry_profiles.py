"""
@author: Brianna Hinds
Description: Build telemetry_profiles.csv from lap_telemetry_df.csv.

Run this ONCE after dataset_build.py finishes.
The output is read by helpers.py at simulation time so the pit-wall
graphs show real (historical median) telemetry instead of synthetic noise.

Output: data/telemetry_profiles.csv
Columns:
  track, compound, tire_age_bucket,
  avg_speed_kmh, min_speed_kmh, max_speed_kmh,
  throttle_percent, braking_percent, coasting_percent,
  avg_gear, gear_changes, drs_pct

tire_age_bucket values: "fresh" (0-5), "prime" (6-15), "used" (16-25),
                        "old"   (26-35), "cliff" (36+)
"""
import os
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")


def _out(f): 
    return os.path.join(_DATA, f)

def _inp(f): 
    return os.path.join(_DATA, f)


TELEM_COLS = [
    "avg_speed_kmh", "min_speed_kmh", "max_speed_kmh",
    "throttle_percent", "braking_percent", "coasting_percent",
    "avg_gear", "gear_changes"
]
BUCKET_EDGES   = [0, 5, 15, 25, 35, 999]
BUCKET_LABELS  = ["fresh", "prime", "used", "old", "cliff"]

# Wet-condition speed scaling derived from 2024 Canadian GP data
# (only track with dry + intermediate + wet laps in same race)
#   dry median:   193.3 km/h
#   inter median: 173.8 km/h  → 0.899
#   wet median:   163.1 km/h  → 0.844
_WET_SCALE: dict[str, dict] = {
    "INTERMEDIATE": {
        "speed":    0.899,   # avg/max/min speed scaled down
        "throttle": 0.88,    # less full-throttle in wet
        "braking":  1.18,    # more braking / later braking points
        "coasting": 1.10,    # more lift-and-coast
        "gear":     0.97,    # marginally lower average gear
        "shifts":   0.95,
    },
    "WET": {
        "speed":    0.844,
        "throttle": 0.75,
        "braking":  1.30,
        "coasting": 1.20,
        "gear":     0.93,
        "shifts":   0.90,
    },
}

def _apply_wet_scale(base: dict, compound: str) -> dict:
    """
    Given a dry-compound profile dict, return a scaled version
    appropriate for INTERMEDIATE or WET conditions.
    """
    s = _WET_SCALE[compound]
    result = {}
    for col in TELEM_COLS:
        v = base.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            result[col] = np.nan
            continue
        if col in ("avg_speed_kmh", "min_speed_kmh", "max_speed_kmh"):
            result[col] = v * s["speed"]
        elif col == "throttle_percent":
            result[col] = min(v * s["throttle"], 1.0)
        elif col == "braking_percent":
            result[col] = min(v * s["braking"], 1.0)
        elif col == "coasting_percent":
            result[col] = min(v * s["coasting"], 1.0)
        elif col == "avg_gear":
            result[col] = v * s["gear"]
        elif col == "gear_changes":
            result[col] = v * s["shifts"]
        else:
            result[col] = v
    return result

def build_profiles(tel_path: str | None = None) -> pd.DataFrame:
    """
    Read lap_telemetry_df.csv, group by (track, compound, tire_age_bucket),
    and take the median of all telemetry features.

    Args
        tel_path : override path to lap_telemetry_df.csv

    Returns
        profiles DataFrame, also saved to data/telemetry_profiles.csv
    """
    path = tel_path or _inp("lap_telemetry_df.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Telemetry data not found at {path}.\n"
            "Run dataset_build.py first."
        )

    print(f"Reading {path} …")
    tel = pd.read_csv(path)
    print(f"  {len(tel):,} rows loaded.")

    # drop inlaps / outlaps — their telemetry is distorted by the pit-lane
    tel = tel[~tel["is_inlap"] & ~tel["is_outlap"]]
    tel = tel[tel["avg_speed_kmh"] > 0]               # zero-speed laps = pit lane / red flag
    tel = tel[tel["avg_gear"] <= 8]                    # filter FastF1 gear encoding glitch
    tel = tel[tel["gear_changes"] <= 100]              # filter gear encoding glitch

    # drop laps with no telemetry at all
    # tel = tel.dropna(subset=["avg_speed_kmh"])

    # normalise compound to uppercase
    tel["compound"] = tel["compound"].str.upper().fillna("UNKNOWN")

    # tire age bucket
    tel["tire_age_bucket"] = pd.cut(
        tel["tire_age"],
        bins=BUCKET_EDGES,
        labels=BUCKET_LABELS,
        right=True,
    ).astype(str)

    # group and aggregate
    group_cols = ["track", "compound", "tire_age_bucket"]
    profiles = (
        tel.groupby(group_cols)[TELEM_COLS]
        .median()
        .reset_index()
    )
    n_real = len(profiles)
    print(f"{n_real} profile rows from real data.")

    # ── fill any (track, compound, bucket) combos that have no real data ──
    # by interpolating from the same track/compound using adjacent buckets.
    # This avoids NaN lookups in the simulator for rare combinations.
    all_combos = pd.MultiIndex.from_product(
        [profiles["track"].unique(),
         profiles["compound"].unique(),
         BUCKET_LABELS],
        names=group_cols,
    ).to_frame(index=False)

    profiles = all_combos.merge(profiles, on=group_cols, how="left")

    # forward/back fill within each (track, compound) group across buckets
    profiles = profiles.sort_values(["track", "compound", "tire_age_bucket"])
    profiles[TELEM_COLS] = (
        profiles.groupby(["track", "compound"])[TELEM_COLS]
        .transform(lambda s: s.ffill().bfill())
    )

    # ── 6. wet / intermediate fallback ────────────────────────────────────
    # Any track × {INTERMEDIATE, WET} rows still all-NaN after step 5 had
    # no real wet laps at that circuit. Derive from the HARD compound profile
    # (most conservative dry baseline) scaled by _WET_SCALE.
    for wet_cmp in ("INTERMEDIATE", "WET"):
        mask_wet = (
            (profiles["compound"] == wet_cmp) &
            profiles["avg_speed_kmh"].isnull()
        )
        if not mask_wet.any():
            continue

        missing_tracks = profiles.loc[mask_wet, "track"].unique()
        print(f"  Deriving {wet_cmp} profiles for {len(missing_tracks)} tracks "
              f"using HARD-compound scaling …")

        for track in missing_tracks:
            # get the HARD profile rows for this track
            hard_rows = profiles[
                (profiles["track"]    == track) &
                (profiles["compound"] == "HARD")
            ]
            if hard_rows.empty:
                # fall back to any available dry compound
                hard_rows = profiles[
                    (profiles["track"]    == track) &
                    (profiles["compound"].isin(["MEDIUM", "SOFT"])) &
                    profiles["avg_speed_kmh"].notna()
                ]
            if hard_rows.empty:
                continue

            for bucket in BUCKET_LABELS:
                # base profile for this bucket (or nearest non-NaN)
                base_row = hard_rows[hard_rows["tire_age_bucket"] == bucket]
                if base_row.empty or base_row[TELEM_COLS].isnull().all(axis=None):
                    base_row = hard_rows.dropna(subset=["avg_speed_kmh"])
                if base_row.empty:
                    continue

                base_dict = base_row[TELEM_COLS].median().to_dict()
                scaled    = _apply_wet_scale(base_dict, wet_cmp)

                for col, val in scaled.items():
                    profiles.loc[
                        (profiles["track"]           == track) &
                        (profiles["compound"]        == wet_cmp) &
                        (profiles["tire_age_bucket"] == bucket),
                        col,
                    ] = val

    # ── 7. final safety net — any remaining NaN → track-level median ──────
    remaining_nan = profiles[profiles["avg_speed_kmh"].isnull()]
    if len(remaining_nan):
        print(f"  {len(remaining_nan)} rows still NaN after wet scaling — "
              f"filling with track-level median.")
        track_medians = (
            profiles.groupby("track")[TELEM_COLS]
            .median()
            .reset_index()
            .rename(columns={c: f"_med_{c}" for c in TELEM_COLS})
        )
        profiles = profiles.merge(track_medians, on="track", how="left")
        for col in TELEM_COLS:
            profiles[col] = profiles[col].fillna(profiles[f"_med_{col}"])
        profiles = profiles.drop(columns=[f"_med_{c}" for c in TELEM_COLS])

    nan_remaining = profiles[TELEM_COLS].isnull().sum().sum()
    print(f"  NaN remaining after all fill strategies: {nan_remaining}")

    # export final profile
    out_path = _out("telemetry_profiles.csv")
    os.makedirs(_DATA, exist_ok=True)
    profiles.to_csv(out_path, index=False)
    print(f"\nProfiles written: {len(profiles)} rows -> {out_path}")
    print(f"  Covers {profiles['track'].nunique()} tracks, "
          f"{profiles['compound'].nunique()} compounds, "
          f"{profiles['tire_age_bucket'].nunique()} age buckets.")
    return profiles


def lookup_profile(
    profiles: pd.DataFrame,
    track: str,
    compound: str,
    tire_age: int | float,
) -> dict:
    """
    Return the telemetry profile dict for a given (track, compound, tire_age).

    Fallback chain (each step only used if the previous returns NaN rows):
      1. exact (track, compound, bucket)
      2. same (track, compound), nearest bucket with real data
      3. same track, any dry compound, same bucket
      4. track median across all compounds + buckets
      5. zeros (should never reach here after build_profiles fills everything)

    Args
        profiles : DataFrame from build_profiles() or load_and_normalise()
        track    : circuit name
        compound : SOFT / MEDIUM / HARD / INTERMEDIATE / WET
        tire_age : current tire age in laps
    """
    # normalise column names defensively — handles CSVs built before the rename
    bucket = BUCKET_LABELS[-1]
    for i, label in enumerate(BUCKET_LABELS):
        if BUCKET_EDGES[i] < tire_age <= BUCKET_EDGES[i + 1]:
            bucket = label
            break

    def _row_to_dict(row: pd.DataFrame) -> dict | None:
        """Return dict only if all TELEM_COLS are non-NaN."""
        if row.empty:
            return None
        med = row[TELEM_COLS].median()
        if med.isnull().any():
            return None
        return med.to_dict()

    # 1. exact match
    result = _row_to_dict(profiles[
        (profiles["track"]           == track) &
        (profiles["compound"]        == compound.upper()) &
        (profiles["tire_age_bucket"] == bucket)
    ])
    if result:
        return result

    # 2. same compound, any bucket
    result = _row_to_dict(
        profiles[
            (profiles["track"]    == track) &
            (profiles["compound"] == compound.upper()) &
            profiles["avg_speed_kmh"].notna()
        ]
    )
    if result:
        return result

    # 3. dry fallback at same track + bucket
    result = _row_to_dict(profiles[
        (profiles["track"]           == track) &
        (profiles["compound"].isin(["HARD", "MEDIUM", "SOFT"])) &
        (profiles["tire_age_bucket"] == bucket) &
        profiles["avg_speed_kmh"].notna()
    ])
    if result:
        return result

    # 4. track median (all compounds, all buckets)
    result = _row_to_dict(profiles[
        (profiles["track"] == track) &
        profiles["avg_speed_kmh"].notna()
    ])
    if result:
        return result

    # 5. absolute last resort
    return {c: 0.0 for c in TELEM_COLS}

if __name__ == "__main__":
    # build_profiles()
    # Also accepts an explicit path for testing against the uploaded files
    import sys
    tel_path = sys.argv[1] if len(sys.argv) > 1 else None
    profiles = build_profiles(tel_path)

    # quick sanity check
    nan_count = profiles[TELEM_COLS].isnull().sum().sum()
    print(f"\nSanity check: {nan_count} NaN values remaining (should be 0)")
    print("\nSample INTERMEDIATE profile (Abu Dhabi):")
    print(profiles[
        (profiles["track"] == "Abu Dhabi Grand Prix") &
        (profiles["compound"] == "INTERMEDIATE") &
        (profiles["tire_age_bucket"] == "fresh")
    ][TELEM_COLS].to_string())