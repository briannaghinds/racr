## Racr: F1 Race Simulator

**Project Scope**: End-to-end MLOPs system that simulates Formula 1 (F1) race outcomes under varying pit stop strategies and recommends optimal strategies based on track, tire, and race conditions. This is deployed in an interactive strategy using Streamlit.

### High-Level System Architecture
```
┌────────────┐
│ Data Layer │  ← historical / synthetic F1 data
└─────┬──────┘
      ↓
┌────────────┐
│ Simulator  │  ← lap-by-lap race simulation
└─────┬──────┘
      ↓
┌────────────┐
│ ML Models  │  ← lap time, tire deg, pit loss
└─────┬──────┘
      ↓
┌────────────┐
│ Optimizer  │  ← strategy search / recommendation
└─────┬──────┘
      ↓
┌────────────┐
│ Streamlit  │  ← user-facing app
└────────────┘
```

### Data
The F1 data will come from the Python library `FastF1`.

### Different Models
- (1) Lap Time Model: XGBoost (inputs: track, compound, tire age, temp, etc)
      - MVP1: Streamlit application that has the user choices (track, compound, etc.) and based on changed inputs will have a delta time (red=slower/green=faster) the base time will be the average time calculated by taking the 2024 data I have and calculating the average value. I won't use the model for base case because the model's purpose is to predict strategy. This helps with preventing overfitting and the outward reasoning makes sense to a non-technical person.
- (2) Pit Stop Simulation: fixed pit loss (track-specific), when someone changes tires that resets the degradation
      - deterministic pit loos
      - tire reset logic
- (3) Safety Car Logic (OPTIONAL FOR NOW): see the affect of the XGBoost model when a saftey car is RANDOMLY deployed, a pit under SC = reduced pit loss
      - stochastic
      - pit advantage modeling

### Project Optimization (Bayesian Optimization)
This is the nitty gritty of what I am doing.
Define the strategies as paramterized objects (like a Grid Search), first do visualizations per model (seaborn pairplot, etc.)

### Dashboard
Each track (dropdown) will be visualized in streamlit and the strategy inputs will be something like radio buttons, dropdowns, etc.

MVP1 -> Lap Time Predictor