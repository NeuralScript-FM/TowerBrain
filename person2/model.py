
"""
TOWER BRAIN | PERSON 2: AI OCCUPANCY PREDICTOR

Configuration:
- 11 floors: F1-F11
- 15-minute simulation interval
- 1-hour prediction horizon (4 intervals)
- 90 days of simulated historical occupancy
- 24-hour timetable format
- Temperature in Celsius
- RandomForestRegressor (scikit-learn)

Run:
    python model.py

Input:
    timetable.csv (optional)

Accepted timetable columns:
    room_id, floor_id, day, start_time, end_time, class_size

Also accepts:
    start_time_24hr, end_time_24hr

Eight core outputs:
    synthetic_occupancy.csv
    prediction_results.csv
    accuracy_report.json
    building_total_predictions.csv
    actual_vs_predicted.png
    feature_importance.csv
    feature_importance.png
    random_forest_model.joblib

All occupancy data is simulated, not real sensor data.
"""


"""
MODEL EVALUATION METRICS - EXPLANATION

MAE (Mean Absolute Error):
    Average absolute difference between predicted and actual occupancy.
    Lower MAE means predictions are closer to actual occupancy.
    Example: MAE = 5 means the model is wrong by about 5 people on average.

RMSE (Root Mean Squared Error):
    Measures prediction error while penalizing large errors more heavily.
    Lower RMSE is better.
    A much higher RMSE than MAE may indicate some large prediction errors.

R2 (R-squared / Coefficient of Determination):
    Measures how well the model explains variations in actual occupancy.
    R2 = 1.0  -> Perfect predictions.
    R2 = 0.0  -> No better than predicting the test-set average.
    R2 < 0.0  -> Worse than predicting the test-set average.
    Closer to 1 is generally better.

ACTUAL:
    Known occupancy value used to evaluate the prediction.
    Currently generated using synthetic simulation data.

PREDICTED:
    Occupancy estimated by the trained Random Forest model.

ABSOLUTE ERROR:
    Absolute difference between actual and predicted occupancy.

FEATURE IMPORTANCE:
    Indicates how useful each input feature was to the model's
    predictions. It does not establish cause and effect.

TRAINING DATA:
    Historical data used by the model to learn patterns.

TESTING DATA:
    Separate data used to evaluate model performance on unseen examples.

GROUND TRUTH:
    Reference actual occupancy used to evaluate predictions.
"""


from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_DAYS = 90
N_FLOORS = 11

INTERVAL_MINUTES = 15
STEPS_PER_HOUR = 60 // INTERVAL_MINUTES
SLOTS_PER_DAY = 24 * 60 // INTERVAL_MINUTES

START_DATE = "2026-08-03 00:00:00"

BASE_DIR = Path(__file__).resolve().parent
TIMETABLE_FILE = BASE_DIR / "timetable.csv"
OUTPUT_DIR = BASE_DIR

DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

REQUIRED_COLUMNS = [
    "room_id",
    "floor_id",
    "day",
    "start_time",
    "end_time",
    "class_size",
]

FEATURE_COLUMNS = [
    "floor_number",
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "future_hour",
    "future_minute",
    "future_day_of_week",
    "current_occupancy",
    "occupancy_lag_15m",
    "occupancy_lag_30m",
    "occupancy_lag_60m",
    "occupancy_same_time_previous_day",
    "occupancy_mean_last_hour",
    "scheduled_occupancy",
    "scheduled_next_hour",
    "outdoor_temp_c",
]


# ============================================================
# 2. TIMETABLE CREATION AND VALIDATION
# ============================================================

def create_demo_timetable():
    """
    Creates a placeholder timetable for testing.

    This is NOT Person 1's real timetable.
    It creates sample classes across weekdays,
    with class times between 09:00 and 22:30.
    """

    rows = []

    start_times = [
        "09:00", "10:30", "12:00",
        "13:30", "15:00", "16:30",
        "18:00", "19:30", "21:00",
    ]

    for floor_num in range(1, N_FLOORS + 1):

        floor_id = f"F{floor_num}"

        for room_num in range(1, 4):

            room_id = f"R{floor_num}{room_num:02d}"

            for day_index, day in enumerate(DAYS[:5]):

                slot_index = (
                    floor_num + room_num + day_index
                ) % len(start_times)

                start = start_times[slot_index]

                start_hour, start_minute = map(
                    int, start.split(":")
                )

                end_total = start_hour * 60 + start_minute + 90

                end = (
                    f"{end_total // 60:02d}:"
                    f"{end_total % 60:02d}"
                )

                class_size = (
                    20 + (
                        floor_num * 13
                        + room_num * 7
                        + day_index * 5
                    ) % 36
                )

                rows.append({
                    "room_id": room_id,
                    "floor_id": floor_id,
                    "day": day,
                    "start_time": start,
                    "end_time": end,
                    "class_size": class_size,
                })

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def load_timetable():

    if TIMETABLE_FILE.exists():

        timetable = pd.read_csv(TIMETABLE_FILE)

        # Support both naming conventions.
        aliases = {
            "start_time_24hr": "start_time",
            "end_time_24hr": "end_time",
        }

        timetable = timetable.rename(columns=aliases)

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in timetable.columns
        ]

        if missing:
            raise ValueError(
                "timetable.csv is missing columns: "
                + ", ".join(missing)
            )

        timetable = timetable[REQUIRED_COLUMNS].copy()

        print(f"Using actual timetable: {TIMETABLE_FILE}")

    else:

        timetable = create_demo_timetable()

        timetable.to_csv(
            OUTPUT_DIR / "timetable_template.csv",
            index=False
        )

        print("\nWARNING: timetable.csv was not found.")
        print("A DEMO timetable has been created.")
        print("Replace it with Person 1's actual timetable later.\n")

    # Clean values.
    timetable["room_id"] = (
        timetable["room_id"].astype(str).str.strip()
    )

    timetable["floor_id"] = (
        timetable["floor_id"].astype(str).str.strip().str.upper()
    )

    timetable["day"] = (
        timetable["day"].astype(str).str.strip().str.capitalize()
    )

    timetable["class_size"] = pd.to_numeric(
        timetable["class_size"],
        errors="coerce"
    )

    if timetable["class_size"].isna().any():
        raise ValueError("class_size contains invalid values.")

    if (timetable["class_size"] < 0).any():
        raise ValueError("class_size cannot be negative.")

    if (timetable["class_size"] > 250).any():
        raise ValueError(
            "A class_size exceeds 250. Check the timetable values."
        )

    timetable["class_size"] = (
        timetable["class_size"].round().astype(int)
    )

    # Validate floors.
    valid_floors = {
        f"F{i}" for i in range(1, N_FLOORS + 1)
    }

    invalid_floors = (
        set(timetable["floor_id"]) - valid_floors
    )

    if invalid_floors:
        raise ValueError(
            f"Invalid floor IDs: {sorted(invalid_floors)}"
        )

    # Validate days.
    invalid_days = set(timetable["day"]) - set(DAYS)

    if invalid_days:
        raise ValueError(
            f"Invalid day names: {sorted(invalid_days)}"
        )

    # Validate time format.
    for column in ["start_time", "end_time"]:

        parsed = pd.to_datetime(
            timetable[column].astype(str).str.strip(),
            format="%H:%M",
            errors="coerce"
        )

        if parsed.isna().any():
            raise ValueError(
                f"{column} must use HH:MM format, "
                "for example 09:00 or 22:30."
            )

        timetable[column] = parsed.dt.strftime("%H:%M")

    # Validate duration and 15-minute alignment.
    for row in timetable.itertuples(index=False):

        start = (
            int(row.start_time[:2]) * 60
            + int(row.start_time[3:])
        )

        end = (
            int(row.end_time[:2]) * 60
            + int(row.end_time[3:])
        )

        if end <= start:
            raise ValueError(
                f"Invalid class duration for room {row.room_id} "
                f"on {row.day}: {row.start_time}-{row.end_time}"
            )

        if start % INTERVAL_MINUTES != 0 or end % INTERVAL_MINUTES != 0:
            raise ValueError(
                f"Class times must align with 15-minute intervals: "
                f"{row.start_time}-{row.end_time}"
            )

    # Check for overlapping classes in the same room on the same day.
    for (room, day), group in timetable.groupby(["room_id", "day"]):

        intervals = []

        for row in group.itertuples(index=False):

            start = (
                int(row.start_time[:2]) * 60
                + int(row.start_time[3:])
            )

            end = (
                int(row.end_time[:2]) * 60
                + int(row.end_time[3:])
            )

            intervals.append((start, end))

        intervals.sort()

        for previous, current in zip(intervals, intervals[1:]):

            if current[0] < previous[1]:
                raise ValueError(
                    f"Overlapping classes found in room {room} "
                    f"on {day}."
                )

    if timetable.empty:
        raise ValueError("The timetable contains no classes.")

    return timetable


# ============================================================
# 3. BUILD TIMETABLE LOOKUP
# ============================================================

def build_schedule_lookup(timetable):

    """
    Creates a lookup of scheduled class populations.

    Key:
        (day, floor_id, minute_of_day)

    Value:
        Active classes at that time.
    """

    lookup = {}

    for row in timetable.itertuples(index=False):

        start = (
            int(row.start_time[:2]) * 60
            + int(row.start_time[3:])
        )

        end = (
            int(row.end_time[:2]) * 60
            + int(row.end_time[3:])
        )

        class_key = (
            row.room_id,
            row.floor_id,
            row.day,
            row.start_time,
            row.end_time,
        )

        for minute in range(
            start,
            end,
            INTERVAL_MINUTES
        ):

            key = (
                row.day,
                row.floor_id,
                minute
            )

            lookup.setdefault(key, []).append(
                (class_key, int(row.class_size))
            )

    return lookup


# ============================================================
# 4. TEMPERATURE SIMULATION
# ============================================================

def outdoor_temperature(ts, rng):

    """
    Generates illustrative outdoor temperature in Celsius.

    This is simulated temperature, not live weather data.
    """

    decimal_hour = ts.hour + ts.minute / 60

    daily_curve = 6.0 * np.sin(
        2 * np.pi * (decimal_hour - 9) / 24
    )

    seasonal_curve = 1.5 * np.sin(
        2 * np.pi * ts.dayofyear / 365
    )

    temperature = (
        35.0
        + daily_curve
        + seasonal_curve
        + rng.normal(0, 0.6)
    )

    return round(float(temperature), 2)


# ============================================================
# 5. GENERATE SYNTHETIC OCCUPANCY HISTORY
# ============================================================

def generate_history(timetable, days=N_DAYS):

    """
    Generates occupancy records for all 11 floors.

    Each floor receives one record every 15 minutes.

    Class attendance remains consistent during a class session.
    """

    rng = np.random.default_rng()

    timestamps = pd.date_range(
        START_DATE,
        periods=days * SLOTS_PER_DAY,
        freq=f"{INTERVAL_MINUTES}min"
    )

    schedule_lookup = build_schedule_lookup(timetable)

    session_attendance = {}

    previous_by_floor = {
        f"F{i}": 0
        for i in range(1, N_FLOORS + 1)
    }

    rows = []

    print(f"Generating {days} days of simulated occupancy...")

    for ts in timestamps:

        temperature = outdoor_temperature(ts, rng)

        minute_of_day = ts.hour * 60 + ts.minute

        day_name = ts.day_name()

        for floor_num in range(1, N_FLOORS + 1):

            floor_id = f"F{floor_num}"

            active_classes = schedule_lookup.get(
                (day_name, floor_id, minute_of_day),
                []
            )

            scheduled = sum(
                size for _, size in active_classes
            )

            expected_attendance = 0.0

            for class_key, class_size in active_classes:

                session_key = (
                    ts.date(),
                ) + class_key

                if session_key not in session_attendance:

                    attendance_rate = rng.beta(8, 2)

                    # Occasionally simulate a poorly attended class.
                    if rng.random() < 0.07:
                        attendance_rate *= rng.uniform(0.0, 0.15)

                    session_attendance[session_key] = attendance_rate

                expected_attendance += (
                    class_size
                    * session_attendance[session_key]
                )

            # Small background occupancy outside class attendance.
            if ts.dayofweek < 5 and 7 <= ts.hour <= 19:
                background = rng.poisson(2)
            else:
                background = rng.poisson(0.5)

            raw_occupancy = max(
                0,
                expected_attendance
                + background
                + rng.normal(0, 2.0)
            )

            previous = previous_by_floor[floor_id]

            # Smooth transitions between 15-minute readings.
            occupancy = int(round(
                0.85 * raw_occupancy
                + 0.15 * previous
            ))

            occupancy = max(0, min(250, occupancy))

            previous_by_floor[floor_id] = occupancy

            rows.append({
                "timestamp": ts,
                "floor_id": floor_id,
                "occupancy": occupancy,
                "scheduled_occupancy": scheduled,
                "outdoor_temp_c": temperature,
            })

    history = pd.DataFrame(rows)

    history = history.sort_values(
        ["floor_id", "timestamp"]
    ).reset_index(drop=True)

    return history


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

def make_features(history):

    """
    Creates model inputs.

    Target:
        Occupancy exactly one hour after the current timestamp.

    One hour = four 15-minute intervals.
    """

    parts = []

    for floor_id, group in history.groupby("floor_id"):

        g = group.sort_values("timestamp").copy()

        # Future target timestamp and occupancy.
        g["target_timestamp"] = (
            g["timestamp"].shift(-STEPS_PER_HOUR)
        )

        g["target_next_hour"] = (
            g["occupancy"].shift(-STEPS_PER_HOUR)
        )

        # Current time features.
        g["hour"] = g["timestamp"].dt.hour
        g["minute"] = g["timestamp"].dt.minute
        g["day_of_week"] = g["timestamp"].dt.dayofweek

        g["is_weekend"] = (
            g["day_of_week"] >= 5
        ).astype(int)

        g["floor_number"] = int(floor_id[1:])

        # Time features at the prediction target.
        g["future_hour"] = (
            g["target_timestamp"].dt.hour
        )

        g["future_minute"] = (
            g["target_timestamp"].dt.minute
        )

        g["future_day_of_week"] = (
            g["target_timestamp"].dt.dayofweek
        )

        # Current occupancy and historical readings.
        g["current_occupancy"] = g["occupancy"]

        g["occupancy_lag_15m"] = (
            g["occupancy"].shift(1)
        )

        g["occupancy_lag_30m"] = (
            g["occupancy"].shift(2)
        )

        g["occupancy_lag_60m"] = (
            g["occupancy"].shift(4)
        )

        g["occupancy_same_time_previous_day"] = (
            g["occupancy"].shift(96)
        )

        g["occupancy_mean_last_hour"] = (
            g["occupancy"]
            .rolling(4, min_periods=4)
            .mean()
        )

        # The timetable tells us what classes are scheduled in the future.
        g["scheduled_next_hour"] = (
            g["scheduled_occupancy"].shift(-STEPS_PER_HOUR)
        )

        parts.append(g)

    data = pd.concat(parts, ignore_index=True)

    # Remove rows without enough history or a valid future target.
    data = data.dropna().copy()

    data["target_next_hour"] = (
        data["target_next_hour"].astype(int)
    )

    return data


# ============================================================
# 7. TRAIN AND EVALUATE RANDOM FOREST
# ============================================================

def train_and_evaluate(data):

    print("Splitting data chronologically...")

    unique_times = np.sort(
        data["timestamp"].unique()
    )

    split_index = int(len(unique_times) * 0.80)

    split_time = unique_times[split_index]

    train = data[
        data["timestamp"] < split_time
    ].copy()

    test = data[
        data["timestamp"] >= split_time
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or testing data is empty. "
            "Check the historical data period."
        )

    print(f"Training records: {len(train):,}")
    print(f"Testing records:  {len(test):,}")

    print("Training Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train["target_next_hour"]
    )

    print("Generating test predictions...")

    predictions = model.predict(
        test[FEATURE_COLUMNS]
    )

    predictions = np.rint(
        np.maximum(0, predictions)
    ).astype(int)

    actual = test["target_next_hour"].to_numpy()

    mae = mean_absolute_error(actual, predictions)

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    r2 = r2_score(actual, predictions)

    # Prediction results for every floor.
    results = test[
        ["timestamp", "target_timestamp", "floor_id"]
    ].copy()

    results["actual_occupancy_next_hour"] = actual

    results["predicted_occupancy_next_hour"] = predictions

    results["absolute_error"] = np.abs(
        actual - predictions
    )

    results["prediction_error"] = (
        predictions - actual
    )

    metrics = {
        "model": "RandomForestRegressor",
        "number_of_floors": N_FLOORS,
        "historical_days": N_DAYS,
        "simulation_interval": "15 minutes",
        "prediction_horizon": "1 hour (4 intervals)",
        "training_percentage": 80,
        "testing_percentage": 20,
        "MAE_people": float(mae),
        "RMSE_people": float(rmse),
        "R_squared": float(r2),
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "training_start": str(train["timestamp"].min()),
        "training_end": str(train["timestamp"].max()),
        "testing_start": str(test["timestamp"].min()),
        "testing_end": str(test["timestamp"].max()),
    }

    return model, results, metrics


# ============================================================
# 8. BUILDING-WIDE PREDICTIONS
# ============================================================

def create_building_totals(results):

    """
    Combines the 11 floor predictions into building-wide totals.

    Each row represents one prediction timestamp.
    """

    totals = (
        results.groupby(
            ["timestamp", "target_timestamp"],
            as_index=False
        )
        .agg(
            actual_building_occupancy=(
                "actual_occupancy_next_hour", "sum"
            ),
            predicted_building_occupancy=(
                "predicted_occupancy_next_hour", "sum"
            ),
        )
    )

    totals["absolute_error"] = np.abs(
        totals["actual_building_occupancy"]
        - totals["predicted_building_occupancy"]
    )

    totals["prediction_error"] = (
        totals["predicted_building_occupancy"]
        - totals["actual_building_occupancy"]
    )

    return totals


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

def create_feature_importance(model):

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    return importance


# ============================================================
# 10. SAVE ALL EIGHT OUTPUT FILES
# ============================================================

def save_outputs(
    history,
    results,
    building_totals,
    metrics,
    model,
    importance,
    timetable,
):

    print("\nSaving output files...")

    # 1. Historical occupancy data.
    history.to_csv(
        OUTPUT_DIR / "synthetic_occupancy.csv",
        index=False
    )

    # 2. Floor-by-floor prediction results.
    results.to_csv(
        OUTPUT_DIR / "prediction_results.csv",
        index=False
    )

    # 3. Accuracy report.
    with open(
        OUTPUT_DIR / "accuracy_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # 4. Building-wide predictions.
    building_totals.to_csv(
        OUTPUT_DIR / "building_total_predictions.csv",
        index=False
    )

    # 5. Actual vs predicted graph.
    floor_sample = (
        results[results["floor_id"] == "F6"]
        .sort_values("timestamp")
        .head(48)
    )

    plt.figure(figsize=(13, 6))

    plt.plot(
        floor_sample["target_timestamp"],
        floor_sample["actual_occupancy_next_hour"],
        label="Actual occupancy",
        linewidth=2
    )

    plt.plot(
        floor_sample["target_timestamp"],
        floor_sample["predicted_occupancy_next_hour"],
        label="Predicted occupancy",
        linewidth=2,
        linestyle="--"
    )

    plt.title(
        "One-Hour-Ahead Occupancy: Floor F6"
    )

    plt.xlabel("Target time")

    plt.ylabel("Number of people")

    plt.xticks(rotation=35)

    plt.legend()

    plt.grid(True, alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "actual_vs_predicted.png",
        dpi=180
    )

    plt.close()

    # 6. Feature importance CSV.
    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    # 7. Feature importance graph.
    plot_data = importance.sort_values(
        "importance",
        ascending=True
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        plot_data["feature"],
        plot_data["importance"]
    )

    plt.title("Random Forest Feature Importance")

    plt.xlabel("Importance")

    plt.ylabel("Input feature")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=180
    )

    plt.close()

    # 8. Save the trained model and its configuration.
    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "timetable": timetable,
        "settings": {
            "number_of_floors": N_FLOORS,
            "interval_minutes": INTERVAL_MINUTES,
            "prediction_horizon_minutes": 60,
            "historical_days": N_DAYS,
            "temperature_unit": "Celsius",
        },
    }

    joblib.dump(
        model_bundle,
        OUTPUT_DIR / "random_forest_model.joblib"
    )

    print("All eight core output files saved.")


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("TOWER BRAIN | AI OCCUPANCY PREDICTOR")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"Floors: {N_FLOORS}")
    print(f"Simulation interval: {INTERVAL_MINUTES} minutes")
    print(f"Prediction horizon: {STEPS_PER_HOUR} intervals (1 hour)")
    print(f"Historical period: {N_DAYS} days")

    # Load actual or demo timetable.
    print("\n[1/6] Loading timetable...")

    timetable = load_timetable()

    print(f"Timetable entries: {len(timetable)}")

    # Generate synthetic historical occupancy.
    print("\n[2/6] Generating occupancy history...")

    history = generate_history(timetable)

    print(f"Generated occupancy records: {len(history):,}")

    print(
        "Expected records:",
        N_DAYS * SLOTS_PER_DAY * N_FLOORS
    )

    # Create model features.
    print("\n[3/6] Creating prediction features...")

    data = make_features(history)

    print(f"Usable feature records: {len(data):,}")

    # Train and evaluate.
    print("\n[4/6] Training and evaluating model...")

    model, results, metrics = train_and_evaluate(data)

    # Create building-wide totals and feature importance.
    print("\n[5/6] Preparing analysis outputs...")

    building_totals = create_building_totals(results)

    importance = create_feature_importance(model)

    # Save all outputs.
    print("\n[6/6] Saving files...")

    save_outputs(
        history,
        results,
        building_totals,
        metrics,
        model,
        importance,
        timetable,
    )

    # Print accuracy.
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"MAE: {metrics['MAE_people']:.3f} people")
    print(f"RMSE: {metrics['RMSE_people']:.3f} people")
    print(f"R²: {metrics['R_squared']:.4f}")

    print("\nMost important features:")

    print(
        importance.head(5).to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)

    output_files = [
        "synthetic_occupancy.csv",
        "prediction_results.csv",
        "accuracy_report.json",
        "building_total_predictions.csv",
        "actual_vs_predicted.png",
        "feature_importance.csv",
        "feature_importance.png",
        "random_forest_model.joblib",
    ]

    for number, filename in enumerate(output_files, start=1):

        path = OUTPUT_DIR / filename

        status = "CREATED" if path.exists() else "MISSING"

        print(f"{number}. {filename} [{status}]")

    print(f"\nSaved in: {OUTPUT_DIR}")

    print("\nTraining pipeline completed.")


if __name__ == "__main__":
    main()
