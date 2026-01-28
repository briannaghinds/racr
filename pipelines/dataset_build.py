"""
@author: Brianna Hinds
Description: Methods for Data Table building.
"""
import fastf1
import pandas as pd

def initialize_data(year: int) -> pd.DataFrame:
    """
    Pulls all race session information for a specific season.
    Initializes all data from the API library.
    Returns concatenated laps dataframe for all races.

    Args
        year: specified season year
    """

    # loop over all races
    # fastf1.Cache.enable_cache("cache/fastf1")
    all_laps = []
    schedule = fastf1.get_event_schedule(year)

    for _, event in schedule.iterrows():
        round_no = event["RoundNumber"]
        race_name = event["EventName"]

        try:
            session = fastf1.get_session(year, round_no, "R")
            session.load(laps=True, weather=True)

            laps = session.laps.copy()
            laps["year"] = year
            laps["round"] = round_no
            laps["track"] = race_name
            laps["race_id"] = str(year) + str(round_no) + str(race_name.replace(" ", "_"))

            all_laps.append(laps)
            print(f"Loaded {race_name}")

        except Exception as e:
            print(f"Failed {race_name}: {e}")

    laps = pd.concat(all_laps, ignore_index=True)
    return laps


def lap_times_data(laps: pd.DataFrame):
    """
    Returns lap_times table pulled from the combined laps dataframe.

    Args
        laps: concatenated laps dataframe
    """
    # lap times dataframe
    lap_times_df = laps[["race_id", "year", "round", "track", "Driver", "Team", "LapNumber", "Stint", "Compound", "TyreLife","LapTime", "PitInTime", "PitOutTime"]].copy()
    lap_times_df = lap_times_df.rename(columns={
        "Driver": "driver",
        "Team": "team",
        "LapNumber": "lap_number",
        "Stint": "stint",
        "Compound": "compound",
        "TyreLife": "tyre_age",
        "LapTime": "lap_time"
        })

    # convert lap time to seconds
    lap_times_df["lap_time_sec"] = lap_times_df["lap_time"].dt.total_seconds()

    # flag pit laps
    lap_times_df["is_inlap"] = lap_times_df["PitInTime"].notna()
    lap_times_df["is_outlap"] = lap_times_df["PitOutTime"].notna()

    # drop helper columns
    lap_times_df = lap_times_df.drop(columns=["lap_time", "PitInTime", "PitOutTime"])

    return lap_times_df 


def race_data(laps: pd.DataFrame):
    """
    Returns races table (one row per race).

    Args
        laps: concatenated laps dataframe
    """
    races_df = laps.groupby("race_id").agg(
        year=("year", "first"),
        round=("round", "first"),
        track=("track", "first"),
        total_laps=("LapNumber", "max")
    ).reset_index()

    # add pit_loss_sec and lap_length_km (if known) MIGHT NEED TO DO THIS MANUALLY
    races_df["pit_loss_sec"] = None
    races_df["lap_length_km"] = None

    return races_df

def driver_data(laps: pd.DataFrame) -> pd.DataFrame:
    drivers_df = (
        laps[["Driver", "Team"]].drop_duplicates().rename(
            columns={
                "Driver": "driver", 
                "Team": "team"
            }
        ).reset_index()
    )

    return drivers_df

def tire_compounds_data(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Returns tire compounds table (static).
    
    Args:
        laps: concatenated laps dataframe
    """
    compounds = sorted(laps["Compound"].dropna().unique())
    tire_df = pd.DataFrame({
        "compound": compounds,
        "base_grip": [1.0]*len(compounds),   # placeholder, tune later
        "deg_rate": [0.01]*len(compounds),   # placeholder
        "cliff_lap": [20]*len(compounds)     # placeholder
    })
    return tire_df

def race_conditions_data(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a race_condition table per race.

    Args
        laps: concatenated laps dataframe
    """

    weather_cols = ["TrackTemp", "AirTemp", "Rainfall"]

    weather_df = (
        laps[["race_id"] + weather_cols]
        .dropna(subset=weather_cols)
        .groupby("race_id")
        .first()
        .reset_index()
        .rename(columns={
            "TrackTemp": "track_temp",
            "AirTemp": "air_temp",
            "Rainfall": "rain"
        })
    )

    return weather_df



## MAIN
if __name__ == "__main__":
    # initialize data
    laps = initialize_data(2024)  # 2025 seems to have no data
    laps.to_csv("../data/concatenated_laps_df.csv")

    # create lap information
    lap_time_df = lap_times_data(laps)
    lap_time_df.to_csv("../data/lap_time_df.csv")

    # create race table
    races_df = race_data(laps)
    races_df.to_csv("../data/races_df.csv")

    # create tire compounds data
    tire_compounds_df = tire_compounds_data(laps)
    tire_compounds_df.to_csv("../data/tire_compounds_df.csv")

    driver_df = driver_data(laps)
    driver_df.to_csv("../data/drivers_df.csv")

    # create race conditions table
    race_conditions_df = race_conditions_data()
    race_conditions_df.to_csv("./data/race_conditions_df.csv")
