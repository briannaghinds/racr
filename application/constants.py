APP_UI = """
        <style>
        /* Import F1-style font */
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;900&display=swap');
        
        /* Global styles */
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            font-family: 'Rajdhani', sans-serif;
        }
        
        /* Main title styling */
        h1 {
            font-family: 'Orbitron', monospace;
            font-weight: 900;
            font-size: 3.5rem !important;
            background: linear-gradient(90deg, #e10600 0%, #ff1e00 50%, #e10600 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            letter-spacing: 4px;
            text-align: center;
            margin-bottom: 0.5rem !important;
            text-shadow: 0 0 30px rgba(225, 6, 0, 0.3);
        }
        
        /* Subtitle */
        .subtitle {
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.2rem;
            color: #888;
            text-align: center;
            letter-spacing: 2px;
            margin-bottom: 2rem;
            text-transform: uppercase;
        }
        
        /* Section headers */
        h2, h3 {
            font-family: 'Orbitron', monospace;
            color: #e10600;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 700;
        }
        
        /* Dashboard panels */
        .dashboard-panel {
            background: rgba(20, 20, 20, 0.8);
            border: 2px solid #333;
            border-left: 4px solid #e10600;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }
        
        /* Metric cards */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
            border: 2px solid #333;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        div[data-testid="metric-container"] > label {
            font-family: 'Orbitron', monospace;
            font-size: 0.9rem !important;
            color: #888 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        div[data-testid="metric-container"] > div {
            font-family: 'Orbitron', monospace;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #e10600 !important;
        }
        
        /* Selectbox and inputs */
        .stSelectbox, .stNumberInput, .stSlider {
            font-family: 'Rajdhani', sans-serif;
        }
        
        .stSelectbox label, .stNumberInput label, .stSlider label {
            font-family: 'Orbitron', monospace;
            color: #e10600 !important;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 1px;
        }
        
        /* Buttons */
        .stButton button {
            font-family: 'Orbitron', monospace;
            background: linear-gradient(135deg, #e10600 0%, #ff1e00 100%);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.75rem 2rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(225, 6, 0, 0.3);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(225, 6, 0, 0.5);
        }
        
        /* Checkbox */
        .stCheckbox label {
            font-family: 'Rajdhani', sans-serif;
            color: #ccc;
            font-size: 1rem;
        }
        
        /* Toggle */
        .stToggle label {
            font-family: 'Orbitron', monospace;
            color: #e10600 !important;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        /* Telemetry bar */
        .telemetry-bar {
            background: #0a0a0a;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            display: flex;
            justify-content: space-around;
            align-items: center;
        }
        
        .telemetry-item {
            text-align: center;
        }
        
        .telemetry-label {
            font-family: 'Rajdhani', sans-serif;
            color: #888;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .telemetry-value {
            font-family: 'Orbitron', monospace;
            color: #e10600;
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        /* Track status indicator */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff00;
            box-shadow: 0 0 10px #00ff00;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Warning text */
        .warning-text {
            color: #ffaa00;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 600;
        }
        
        /* Stint badge */
        .stint-badge {
            display: inline-block;
            background: #e10600;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-family: 'Orbitron', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 1px;
            margin-right: 0.5rem;
        }
        </style>
    """

PIT_WALL_UI = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        background-color: #080808 !important;
        color: #dddddd;
        font-family: 'Rajdhani', sans-serif;
    }
    .pit-banner {
        background: linear-gradient(90deg, #0d0d0d 0%, #1a0000 50%, #0d0d0d 100%);
        border-bottom: 2px solid #e10600;
        padding: 12px 24px;
        display: flex; align-items: center; gap: 16px;
        margin-bottom: 16px;
    }
    .pit-banner h1 {
        font-family: 'Orbitron', monospace; font-size: 1.6rem;
        font-weight: 900; color: #ffffff; letter-spacing: 4px; margin: 0;
    }
    .pit-banner .sub {
        color: #e10600; font-size: 0.75rem;
        letter-spacing: 6px; font-family: 'Orbitron', monospace;
    }
    .panel {
        background: #0f0f0f; border: 1px solid #222;
        border-radius: 4px; padding: 14px 16px; margin-bottom: 12px;
    }
    .panel-header {
        font-family: 'Orbitron', monospace; font-size: 0.7rem;
        letter-spacing: 3px; color: #e10600; text-transform: uppercase;
        border-bottom: 1px solid #1e1e1e; padding-bottom: 6px; margin-bottom: 10px;
    }
    .stint-badge {
        display: inline-block; background: #e10600; color: #fff;
        font-family: 'Orbitron', monospace; font-size: 0.65rem;
        font-weight: 700; letter-spacing: 2px;
        padding: 2px 8px; border-radius: 2px; margin-bottom: 6px;
    }
    .telem-row   { display: flex; gap: 10px; margin: 8px 0; }
    .telem-tile  {
        flex: 1; background: #141414; border: 1px solid #222;
        border-radius: 4px; padding: 8px 12px; text-align: center;
    }
    .telem-tile .label {
        font-size: 0.6rem; letter-spacing: 2px; color: #666;
        font-family: 'Orbitron', monospace; text-transform: uppercase;
    }
    .telem-tile .value {
        font-size: 1.4rem; font-weight: 700;
        font-family: 'Orbitron', monospace; color: #fff; margin-top: 2px;
    }
    .telem-tile .value.good { color: #44ff88; }
    .telem-tile .value.bad  { color: #ff4444; }
    .telem-tile .value.warn { color: #ffd700; }
    .tire-dot {
        display: inline-block; width: 12px; height: 12px;
        border-radius: 50%; margin-right: 6px; vertical-align: middle;
    }
    .verdict {
        padding: 10px 16px; border-radius: 4px;
        font-family: 'Orbitron', monospace; font-size: 0.85rem;
        font-weight: 700; letter-spacing: 2px; margin-top: 8px;
    }
    .verdict.faster { background: #0d2b1a; border: 1px solid #44ff88; color: #44ff88; }
    .verdict.slower { background: #2b0d0d; border: 1px solid #ff4444; color: #ff4444; }
    .data-badge {
        display: inline-block; font-family: 'Orbitron', monospace;
        font-size: 0.6rem; letter-spacing: 2px; padding: 2px 8px;
        border-radius: 2px; margin-left: 8px; vertical-align: middle;
    }
    .data-badge.real { background: #0d2b1a; border: 1px solid #44ff88; color: #44ff88; }
    .data-badge.sim  { background: #1a1000; border: 1px solid #ffaa00; color: #ffaa00; }
    .divider { height: 1px; background: linear-gradient(90deg, transparent, #333, transparent); margin: 16px 0; }
    </style>
    """

PIT_STOP_LOSS = 22.0

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
INPUT_COLS = ["fuel_effect", "temp_delta", "race_lap", "tire_age", "tire_age_squared", "stint", "track_temp", "air_temp", "is_rain", "track_te", "circuit_length(km)", "compound_HARD", "compound_INTERMEDIATE", "compound_MEDIUM", "compound_SOFT", "compound_WET", "avg_speed_kmh", "min_speed_kmh", "braking_percent", "throttle_percent", "avg_gear", "gear_changes"]

DEFAULT_VALS = {
    "fuel_effect": 0.0,  # assume neutral fuel load 
    "temp_delta": 0.0,  # no temp difference (perfect conditions) 
    "race_lap": 1, 
    "tire_age": 1, 
    "tire_age_squared": 1,  # assume tire age is 1
    "stint": 1,  # first stint
    "track_temp": 30.0,  # average track temp 
    "air_temp": 25.0,  # average air temp 
    "is_rain": 0, 
    "track_te": 92.72733323225049,  # global track mean 
    "circuit_length(km)": 0,  # will pull from dataset
    "compound_HARD": 0,  # compound value gets pulled from user
    "compound_INTERMEDIATE": 0,  
    "compound_MEDIUM": 0, 
    "compound_SOFT": 0, 
    "compound_WET": 0, 
    "avg_speed_kmh": 200.94,  # real median value 
    "min_speed_kmh": 67.0,   # braking zones
    # "max_speed_kmh": 310.0,
    "throttle_percent": 0.49, 
    "braking_percent": 0.21,
    # "coasting_percent": 0.05, 
    "avg_gear": 5.19, 
    "gear_changes": 38
}
# X = final_df[["fuel_effect", 
#               "temp_delta", 
#               "is_inlap", 
#               "is_outlap",
#               "race_lap", 
#               "tire_age", 
#               "tire_age_squared", 
#               "post_cliff", 
#               "stint", 
#               "track_temp", 
#               "air_temp", 
#               "is_rain", 
#               "track_te", 
#               "circuit_length(km)", 
#               "compound_HARD", 
#               "compound_INTERMEDIATE", 
#               "compound_MEDIUM", 
#               "compound_SOFT", 
#               "compound_WET", 
#               "avg_speed_kmh", 
#               "min_speed_kmh",
#               "max_speed_kmh", 
#               "braking_percent", 
#               "throttle_percent", 
#               "coasting_percent", 
#               "avg_gear", 
#               "gear_changes"]]


TIRE_COLORS = {
    "SOFT": "#c1121f",
    "MEDIUM": "#ffd60a",
    "HARD": "#ffffff",
    "WET": "#219ebc",
    "INTERMEDIATE": "#6a994e",
}

TIRE_DEGRAD = {
    "SOFT": 0.12,
    "MEDIUM": 0.07,
    "HARD": 0.04,
    "INTERMEDIATE": 0.05,
    "WET": 0.03
}

CLIFF_LAP = {
    "SOFT": 15,
    "MEDIUM": 25,
    "HARD": 35,
    "INTERMEDIATE": 40,
    "WET": 45
}