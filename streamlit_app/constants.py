PIT_STOP_LOSS = 0.0

TRACKS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix',
       'Australian Grand Prix', 'Japanese Grand Prix',
       'Chinese Grand Prix', 'Miami Grand Prix',
       'Emilia Romagna Grand Prix', 'Monaco Grand Prix',
       'Canadian Grand Prix', 'Spanish Grand Prix', 'Austrian Grand Prix',
       'British Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix',
       'Dutch Grand Prix', 'Italian Grand Prix', 'Azerbaijan Grand Prix',
       'Singapore Grand Prix', 'United States Grand Prix',
       'Mexico City Grand Prix', 'São Paulo Grand Prix',
       'Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix']

# added is_inlap and is_outlap
INPUT_COLS = ["fuel_effect", "temp_delta", "is_inlap", "is_outlap","race_lap", "tire_age", "tire_age_squared", "stint", "track_temp", "air_temp", "is_rain", "track_te", "circuit_length(km)", "compound_HARD", "compound_INTERMEDIATE", "compound_MEDIUM", "compound_SOFT", "compound_WET"]

DEFAULT_VALS = {
    "tire_age_squared": 1,  # assume tire age is 1
    "fuel_effect": 0.0,  # assume neutral fuel load 
    "temp_delta": 0.0,  # no temp difference (perfect conditions) 
    "race_lap": 1, 
    "tire_age": 1, 
    "stint": 1,  # first stint
    "track_temp": 30.0,  # average track temp 
    "air_temp": 25.0,  # average air temp 
    "is_rain": 0, 
    "track_te": 92.72733323225049,  # global track mean 
    "compound_HARD": 0,  # compound value gets pulled from user
    "compound_INTERMEDIATE": 0,  
    "compound_MEDIUM": 0, 
    "compound_SOFT": 0, 
    "compound_WET": 0, 
    "circuit_length(km)": 0  # will pull from dataset
}