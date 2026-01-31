"""
@author: Brianna Hinds
Description: Methods for Data Table building.
"""
import fastf1
import pandas as pd

## DATA INGESTION
def initialize_data(year: int) -> pd.DataFrame:
    """
    Pulls all race session information for a specific season.
    Initializes all data from the API library.
    Returns concatenated laps dataframe for all races.

    Args
        year: specified season year
    """

    # loop over all races in a season
    # NOTE: i don't think 2025 is fully on the fastf1 databases so I am pulling all of 2024
    # fastf1.Cache.enable_cache("cache/fastf1")
    all_laps = []
    all_tracks = []
    schedule = fastf1.get_event_schedule(year)

    for _, event in schedule.iterrows():
        round_no = event["RoundNumber"]
        race_name = event["EventName"]

        try:
            session = fastf1.get_session(year, round_no, "R")
            session.load(laps=True, weather=True)

            laps = session.laps.copy()
            
            # get circuit info
            fast_lap = session.laps.pick_fastest()  # used in pos variable
            pos = fast_lap.get_pos_data()
            circuit_info = session.get_circuit_info()

            race_id = f"{year}_{round_no:02d}_{race_name.replace(' ', '_')}"

            # track line
            pos_df = (pos[["X", "Y"]].dropna().reset_index(drop=True))
            pos_df["seq"] = pos_df.index
            pos_df["type"] = "track"

            # corners
            corners_df = circuit_info.corners.copy()
            corners_df["corner"] = (
                corners_df["Number"].astype(str) +
                corners_df["Letter"].fillna("")
            )
            corners_df = corners_df.rename(columns={"Angle": "angle"})
            corners_df = corners_df[["X", "Y", "corner", "angle"]]
            corners_df["type"] = "corner"
            corners_df["seq"] = None

            # metadata
            for df in (pos_df, corners_df):
                df["race_id"] = race_id
                df["year"] = year
                df["round"] = round_no
                df["track"] = race_name

            track_df = pd.concat([pos_df, corners_df], ignore_index=True)
            all_tracks.append(track_df)

            # pull weather data and merge
            weather = session.weather_data.copy()
            weather = weather.sort_values("Time")
            laps = laps.sort_values("LapStartTime")

            laps = pd.merge_asof(
                laps,
                weather,
                left_on="LapStartTime",
                right_on="Time",
                direction="nearest"
            )

            # laps = session.laps.copy()
            # add metadata after merge
            laps["year"] = year
            laps["round"] = round_no
            laps["track"] = race_name
            laps["race_id"] = race_id

            all_laps.append(laps)
            print(f"Loaded {race_name}")

        except Exception as e:
            print(f"Failed {race_name}: {e}")

    laps = pd.concat(all_laps, ignore_index=True)
    track_info = pd.concat(all_tracks, ignore_index=True)

    return laps, track_info


def lap_times_data(laps: pd.DataFrame) -> pd.DataFrame:
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
        "LapNumber": "race_lap",
        "Stint": "stint",
        "Compound": "compound",
        "TyreLife": "tire_age",
        "LapTime": "lap_time"
        })

    # convert lap time to seconds
    lap_times_df["lap_time_sec"] = lap_times_df["lap_time"].dt.total_seconds()

    # flag pit laps
    lap_times_df["is_inlap"] = lap_times_df["PitInTime"].notna()
    lap_times_df["is_outlap"] = lap_times_df["PitOutTime"].notna()

    # drop helper columns
    lap_times_df = lap_times_df.drop(columns=["lap_time", "PitInTime", "PitOutTime"])

    # ML sanity filter
    lap_times_df = lap_times_df[
        (lap_times_df["lap_time_sec"].notna())
        & (~lap_times_df["is_inlap"])
        & (~lap_times_df["is_outlap"])
    ]

    return lap_times_df 


def race_data(laps: pd.DataFrame) -> pd.DataFrame:
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


def stint_data(lap_times: pd.DataFrame) -> pd.DataFrame:
    stints_df = lap_times.groupby(
        ["race_id", "driver", "stint", "compound", "track"]).agg(
            start_lap=("race_lap", "min"),
            end_lap=("race_lap", "max"),
            stint_length=("race_lap", "count"),
            avg_lap_time=("lap_time_sec", "mean")
    ).reset_index()

    return stints_df


def driver_data(laps: pd.DataFrame) -> pd.DataFrame:
    drivers_df = (
        laps[["Driver", "Team"]].drop_duplicates().rename(
            columns={
                "Driver": "driver", 
                "Team": "team"
            }
        ).reset_index(drop=True)
    )

    return drivers_df


def tire_compounds_data() -> pd.DataFrame:
    """
    Returns tire compounds table (static).
    
    Args:
        laps: concatenated laps dataframe
    """
    # compounds = sorted(laps["Compound"].dropna().unique())
    # tire_df = pd.DataFrame({
    #     "compound": compounds,
    #     "base_grip": [1.0]*len(compounds),   # placeholder, tune later
    #     "deg_rate": [0.01]*len(compounds),   # placeholder
    #     "cliff_lap": [20]*len(compounds)     # placeholder
    # })

    # return tire_df
    return pd.DataFrame({
    "compound": ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
    "base_grip": [1.05, 1.00, 0.97, 0.95, 0.93],
    "deg_rate": [0.025, 0.018, 0.012, 0.010, 0.008],
    "cliff_lap": [15, 25, 35, 40, 45]
    })


def race_conditions_data(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a race_condition table per race.

    Args
        laps: concatenated laps dataframe
    """

    weather_df = (
        laps[["race_id", "TrackTemp", "AirTemp", "Rainfall"]]
        .groupby("race_id")
        .agg({
            "TrackTemp": "mean",
            "AirTemp": "mean",
            "Rainfall": "max"
        })
        .reset_index()
        .rename(columns={
            "TrackTemp": "track_temp",
            "AirTemp": "air_temp",
            "Rainfall": "rain"
        })
    )

    return weather_df


def track_data() -> pd.DataFrame:
    """
    Manually created DataFrame for each circuit.
    Returns DataFrame object of the circuit information.
    """
    # data pulled from https://motorsporttickets.com/blog/how-many-laps-does-each-formula-1-race-have/ and Formula1 website
    # I have all tracks used from 2024-2026 seasons
    data = {
        "track": ["Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix", "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Miami Grand Prix", "Canadian Grand Prix", "Monaco Grand Prix", "Barcelona Grand Prix", "Austrian Grand Prix", "British Grand Prix", "Belgian Grand Prix", "Hungarian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix", "Spanish Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix", "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix", "Emilia Romagna Grand Prix", "Spain Grand Prix"],
        "circuit": ["Melbourne Grand Prix Circuit", "Shanghai International Circuit", "Suzuka", "Bahrain International Circuit", "Jeddah Street Circuit", "Hard Rock Stadium Circuit", "Circuit Gilles Villeneuve", "Circuit de Monaco", "Circuit de Barcelona-Catalunya", "Red Bull Ring", "Silverstone Circuit", "Circuit de Spa-Francorchamps", "Hangaroring", "Zandvoort", "Autodromo Nazionale di Monza", "Madring Circuit", "Baku City Circuit", "Marina Bay Street Circuit", "Circuit of the Americas", "Autodromo Hermanos Rodriguez", "Autodromo Jose Carlos Pace", "Las Vegas Street Circuit", "Lusail Circuit", "Yas Marina Circuit", "Autodromo Enzo e Dino Ferrari", "Circuit de Barcelona-Catalunya"],
        "circuit_length(km)": [5.303, 5.451, 5.807, 5.412, 6.175, 5.41, 4.361, 3.337, 4.655, 4.318, 5.891, 7.004, 4.381, 4.259, 5.793, 5.474, 6.003, 5.063, 5.513, 4.304, 4.309, 6.201, 5.419, 5.554, 4.909, 4.657],
        "race_distance(km)": [307.574, 305.066, 307.471, 308.238, 308.750, 308.37, 305.270, 260.286, 307.104, 306.452, 306.198, 308.052, 306.630, 306.648, 306.720, 312.018, 306.049, 308.706, 308.405, 305.354, 305.879, 310.05, 308.826, 305.355, 309.049, 307.236],
        "laps": [58, 56, 53, 57, 50, 57, 70, 78, 66, 71, 52, 44, 70, 72, 53, 57, 51, 61, 56, 71, 71, 50, 57, 55, 63, 66] 
    }

    track_df = pd.DataFrame(data)

    return track_df




## MAIN
if __name__ == "__main__":
    # initialize data from api
    laps, circuit_info = initialize_data(2024)  # 2025 seems to have no data or either incomplete data

    ## DATAFRAME CREATIONS ##
    # create lap information
    lap_times_df = lap_times_data(laps)

    # create race table
    races_df = race_data(laps)

    # create stint table
    stint_df = stint_data(lap_times_df)

    # create tire compounds data
    tire_compounds_df = tire_compounds_data()

    # create drivers table
    driver_df = driver_data(laps)

    # create race conditions table
    race_conditions_df = race_conditions_data(laps)

    # create track table
    track_df = track_data()

    ## ASSERTIONS (sanity check) ##
    # assert lap_time_df["lap_time_sec"].min() > 60
    # assert lap_time_df["lap_time_sec"].max() < 200

    ## TABLE COMBINATIONS ##
    # combine the track information for each lap_time_df
    lap_times_df = lap_times_df.merge(
        track_df[["track", "circuit_length(km)"]],
        on="track",
        how="left"
    )

## CSV CREATIONS ##
laps.to_csv("../data/concatenated_laps_df.csv", index=False)
circuit_info.to_csv("circuit_info.csv", index=False)
lap_times_df.to_csv("../data/lap_time_df.csv", index=False)
races_df.to_csv("../data/races_df.csv", index=False)
stint_df.to_csv("../data/stint_df.csv", index=False)
tire_compounds_df.to_csv("../data/tire_compounds_df.csv", index=False)
driver_df.to_csv("../data/drivers_df.csv", index=False)
race_conditions_df.to_csv("../data/race_conditions_df.csv", index=False)
track_df.to_csv("../data/track_df.csv", index=False)