from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Jet Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DEEP_LEARNING_DIR = MODELS_DIR / "deep_learning"

MODEL_PATH = DEEP_LEARNING_DIR / "gru_model.keras"


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 20

SENSOR_COLS = [
    f"sensor_{i}"
    for i in range(1, 22)
]

EXPECTED_INPUT_SHAPE = (
    1,
    SEQUENCE_LENGTH,
    len(SENSOR_COLS)
)


# ============================================================
# NASA C-MAPSS COLUMN NAMES
# ============================================================

NASA_COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + SENSOR_COLS
)


# ============================================================
# HELPER: FIND FILE RECURSIVELY
# ============================================================

def find_file(filename):
    """
    Search the complete project directory for a file.
    """

    matches = list(PROJECT_ROOT.rglob(filename))

    if not matches:
        return None

    # Prefer the first existing regular file
    for path in matches:
        if path.is_file():
            return path

    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_gru_model():

    if not MODEL_PATH.exists():
        return None

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    return model


gru_model = load_gru_model()


# ============================================================
# HEADER
# ============================================================

st.title("✈️ Jet Engine Predictive Maintenance")

st.subheader(
    "Remaining Useful Life Prediction"
)

st.write(
    """
    This application uses a trained GRU deep learning model
    to estimate the Remaining Useful Life (RUL) of a jet engine
    from its recent sensor measurements.
    """
)


# ============================================================
# MODEL STATUS
# ============================================================

if gru_model is None:

    st.error(
        "GRU model could not be loaded."
    )

    st.code(
        str(MODEL_PATH)
    )

    st.stop()


st.success(
    "GRU model loaded successfully."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model information"):

    st.write(
        "Model path:",
        str(MODEL_PATH)
    )

    st.write(
        "Model input shape:",
        gru_model.input_shape
    )

    st.write(
        "Model output shape:",
        gru_model.output_shape
    )

    expected_model_shape = (
        None,
        SEQUENCE_LENGTH,
        len(SENSOR_COLS)
    )

    if gru_model.input_shape != expected_model_shape:

        st.warning(
            f"Expected model input shape "
            f"{expected_model_shape}, but found "
            f"{gru_model.input_shape}."
        )


# ============================================================
# LOAD SHAP FILE
# ============================================================

@st.cache_data
def load_shap_data():

    possible_names = [
        "gru_shap_feature_importance.csv",
        "gru_local_shap_explanation.csv",
    ]

    for filename in possible_names:

        path = find_file(filename)

        if path is not None:

            try:

                shap_df = pd.read_csv(path)

                return shap_df, path

            except Exception:
                continue

    return None, None


shap_df, shap_path = load_shap_data()


# ============================================================
# FIND TRAINING DATA
# ============================================================

@st.cache_data
def find_training_data():

    possible_names = [
        "train_FD001.csv",
        "train_FD001.txt",
        "train_FD001"
    ]

    for filename in possible_names:

        path = find_file(filename)

        if path is not None:
            return path

    return None


TRAIN_PATH = find_training_data()


# ============================================================
# CREATE TRAINING SCALER
# ============================================================

@st.cache_resource
def create_training_scaler(train_path_string):

    if train_path_string is None:
        return None

    train_path = Path(train_path_string)

    try:

        # ----------------------------------------------------
        # First attempt: normal CSV with header
        # ----------------------------------------------------

        train_df = pd.read_csv(
            train_path
        )

        # ----------------------------------------------------
        # Detect headerless NASA C-MAPSS format
        # ----------------------------------------------------

        if not all(
            sensor in train_df.columns
            for sensor in SENSOR_COLS
        ):

            train_df = pd.read_csv(
                train_path,
                sep=r"\s+|,",
                engine="python",
                header=None
            )

            # NASA FD001 normally has:
            # unit + cycle + 3 settings + 21 sensors
            if train_df.shape[1] >= 26:

                train_df = train_df.iloc[:, :26].copy()

                train_df.columns = NASA_COLUMNS

            else:

                return None

        # ----------------------------------------------------
        # Make sure sensor columns are numeric
        # ----------------------------------------------------

        sensor_data = train_df[
            SENSOR_COLS
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Remove rows that contain no usable sensor values
        sensor_data = sensor_data.dropna(
            how="all"
        )

        if sensor_data.empty:
            return None

        # Fill any partial missing values
        sensor_data = sensor_data.fillna(
            sensor_data.median()
        )

        scaler = StandardScaler()

        scaler.fit(
            sensor_data.to_numpy(
                dtype=np.float64
            )
        )

        return scaler

    except Exception:
        return None


training_scaler = create_training_scaler(
    str(TRAIN_PATH)
    if TRAIN_PATH is not None
    else None
)


# ============================================================
# SHOW PREPROCESSING STATUS
# ============================================================

with st.expander("Preprocessing information"):

    if TRAIN_PATH is not None:

        st.success(
            "Training dataset found."
        )

        st.write(
            "Training dataset:",
            str(TRAIN_PATH)
        )

    else:

        st.error(
            "train_FD001.csv was not found inside the project."
        )

    if training_scaler is not None:

        st.success(
            "Training StandardScaler created successfully."
        )

        st.write(
            "The same scaler is used to standardize "
            "the uploaded test sensor data before GRU inference."
        )

    else:

        st.error(
            "Training StandardScaler could not be created."
        )

        st.write(
            """
            The GRU was trained on standardized sensor values,
            so the original training dataset is required to
            reproduce the same preprocessing.
            """
        )

        if TRAIN_PATH is None:

            st.write(
                "Expected file: train_FD001.csv"
            )


# ============================================================
# 1. UPLOAD ENGINE SENSOR DATA
# ============================================================

st.header(
    "1. Upload Engine Sensor Data"
)

uploaded_file = st.file_uploader(
    "Upload a CSV containing engine sensor measurements",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Please upload test_FD001.csv to continue."
    )

    st.stop()


# ============================================================
# READ UPLOADED FILE
# ============================================================

try:

    uploaded_file.seek(0)

    test_df = pd.read_csv(
        uploaded_file
    )

except pd.errors.EmptyDataError:

    st.error(
        "The uploaded CSV is empty. "
        "Please upload a valid test_FD001.csv file."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Unable to read the uploaded CSV: {e}"
    )

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

test_df.columns = (
    test_df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# VALIDATE TEST DATA
# ============================================================

required_columns = [
    "unit",
    "cycle",
    *SENSOR_COLS
]

missing_columns = [
    col
    for col in required_columns
    if col not in test_df.columns
]


if missing_columns:

    st.error(
        "The uploaded dataset is missing required columns."
    )

    st.write(
        missing_columns
    )

    st.stop()


# ============================================================
# CONVERT DATA TYPES
# ============================================================

test_df["unit"] = pd.to_numeric(
    test_df["unit"],
    errors="coerce"
)

test_df["cycle"] = pd.to_numeric(
    test_df["cycle"],
    errors="coerce"
)

for sensor in SENSOR_COLS:

    test_df[sensor] = pd.to_numeric(
        test_df[sensor],
        errors="coerce"
    )


# Remove invalid unit/cycle rows
test_df = test_df.dropna(
    subset=["unit", "cycle"]
).copy()


# Sort correctly
test_df = test_df.sort_values(
    ["unit", "cycle"]
).reset_index(
    drop=True
)


if test_df.empty:

    st.error(
        "The uploaded dataset contains no valid engine records."
    )

    st.stop()


st.success(
    f"Successfully loaded {uploaded_file.name}"
)


# ============================================================
# 2. DATASET OVERVIEW
# ============================================================

st.header(
    "2. Dataset Overview"
)

engine_ids = sorted(
    test_df["unit"].unique()
)

total_engines = len(
    engine_ids
)

total_cycles = len(
    test_df
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Engines",
        total_engines
    )


with col2:

    st.metric(
        "Total Cycles",
        total_cycles
    )


with col3:

    st.metric(
        "Sensors",
        len(SENSOR_COLS)
    )


with st.expander(
    "View uploaded data"
):

    st.dataframe(
        test_df.head(20),
        use_container_width=True
    )


# ============================================================
# 3. SELECT ENGINE
# ============================================================

st.header(
    "3. Select Engine"
)

selected_engine = st.selectbox(
    "Choose an engine to analyze:",
    engine_ids
)


# ============================================================
# SELECT ENGINE DATA
# ============================================================

engine_df = test_df[
    test_df["unit"] == selected_engine
].copy()


engine_df = engine_df.sort_values(
    "cycle"
).reset_index(
    drop=True
)


total_engine_cycles = len(
    engine_df
)

latest_cycle = int(
    engine_df["cycle"].max()
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Selected Engine",
        f"Engine {int(selected_engine)}"
    )


with col2:

    st.metric(
        "Latest Cycle",
        latest_cycle
    )


# ============================================================
# CHECK SEQUENCE LENGTH
# ============================================================

if total_engine_cycles < SEQUENCE_LENGTH:

    st.error(
        f"Engine {int(selected_engine)} has only "
        f"{total_engine_cycles} cycles. "
        f"At least {SEQUENCE_LENGTH} cycles are required."
    )

    st.stop()


# ============================================================
# CREATE RAW 20 x 21 SEQUENCE
# ============================================================

raw_sequence_df = (
    engine_df[
        SENSOR_COLS
    ]
    .tail(SEQUENCE_LENGTH)
    .copy()
)


# ============================================================
# HANDLE MISSING SENSOR VALUES
# ============================================================

missing_values = int(
    raw_sequence_df.isna()
    .sum()
    .sum()
)


if missing_values > 0:

    st.warning(
        f"{missing_values} missing sensor values detected. "
        "Missing values will be filled using the training "
        "sensor medians."
    )

    # Use training medians if possible
    if TRAIN_PATH is not None:

        try:

            # Re-read training data for medians
            train_for_median = pd.read_csv(
                TRAIN_PATH
            )

            if not all(
                sensor in train_for_median.columns
                for sensor in SENSOR_COLS
            ):

                train_for_median = pd.read_csv(
                    TRAIN_PATH,
                    sep=r"\s+|,",
                    engine="python",
                    header=None
                )

                train_for_median = (
                    train_for_median
                    .iloc[:, :26]
                    .copy()
                )

                train_for_median.columns = (
                    NASA_COLUMNS
                )

            training_medians = (
                train_for_median[
                    SENSOR_COLS
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
                .median()
            )

            raw_sequence_df = (
                raw_sequence_df
                .fillna(training_medians)
            )

        except Exception:

            raw_sequence_df = (
                raw_sequence_df
                .fillna(0)
            )

    else:

        raw_sequence_df = (
            raw_sequence_df
            .fillna(0)
        )


# ============================================================
# RAW INPUT ARRAY
# ============================================================

X_raw = raw_sequence_df.to_numpy(
    dtype=np.float32
)


if X_raw.shape != (
    SEQUENCE_LENGTH,
    len(SENSOR_COLS)
):

    st.error(
        f"Unexpected raw sequence shape: "
        f"{X_raw.shape}. "
        f"Expected {(20, 21)}."
    )

    st.stop()


# ============================================================
# IMPORTANT:
# STANDARDIZE USING TRAINING SCALER
# ============================================================

if training_scaler is None:

    st.error(
        """
        The GRU cannot safely make a prediction because its
        training StandardScaler is unavailable.

        Put train_FD001.csv somewhere inside the project
        directory and restart Streamlit.
        """
    )

    st.stop()


X_scaled = training_scaler.transform(
    X_raw
)


# ============================================================
# CONVERT TO FLOAT32
# ============================================================

X_scaled = X_scaled.astype(
    np.float32
)


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

X_input = X_scaled.reshape(
    1,
    SEQUENCE_LENGTH,
    len(SENSOR_COLS)
)


# ============================================================
# FINAL INPUT VALIDATION
# ============================================================

if X_input.shape != EXPECTED_INPUT_SHAPE:

    st.error(
        f"Unexpected GRU input shape: "
        f"{X_input.shape}. "
        f"Expected {EXPECTED_INPUT_SHAPE}."
    )

    st.stop()


# ============================================================
# INPUT INFORMATION
# ============================================================

with st.expander(
    "Inference input details"
):

    st.write(
        "Raw input shape:",
        X_raw.shape
    )

    st.write(
        "GRU input shape:",
        X_input.shape
    )

    st.write(
        "Raw input minimum:",
        float(X_raw.min())
    )

    st.write(
        "Raw input maximum:",
        float(X_raw.max())
    )

    st.write(
        "Standardized input minimum:",
        float(X_input.min())
    )

    st.write(
        "Standardized input maximum:",
        float(X_input.max())
    )

    st.write(
        "Standardized input mean:",
        float(X_input.mean())
    )

    recent_cycles = (
        engine_df["cycle"]
        .tail(SEQUENCE_LENGTH)
    )

    st.write(
        f"Prediction uses cycles "
        f"{int(recent_cycles.min())} "
        f"to "
        f"{int(recent_cycles.max())}."
    )


# ============================================================
# 4. GRU PREDICTION
# ============================================================

prediction_output = gru_model.predict(
    X_input,
    verbose=0
)


raw_prediction = float(
    np.asarray(
        prediction_output
    ).flatten()[0]
)


predicted_rul = max(
    0.0,
    raw_prediction
)


# ============================================================
# HEALTH STATUS
# ============================================================

if predicted_rul > 50:

    health_status = "Healthy"

elif predicted_rul > 20:

    health_status = "Degrading"

else:

    health_status = "Critical"


# ============================================================
# PREDICTION RESULT
# ============================================================

st.header(
    "4. Prediction"
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Estimated Remaining Useful Life",
        f"{predicted_rul:.2f} cycles"
    )


with col2:

    st.metric(
        "Health Status",
        health_status
    )


# ============================================================
# PREDICTION INTERPRETATION
# ============================================================

if health_status == "Healthy":

    st.success(
        "The selected engine is currently predicted "
        "to have a relatively high remaining useful life."
    )

elif health_status == "Degrading":

    st.warning(
        "The selected engine shows signs of degradation "
        "and should be monitored."
    )

else:

    st.error(
        "The selected engine is predicted to be near "
        "the end of its useful life."
    )


# ============================================================
# 5. RECENT SENSOR TRENDS
# ============================================================

st.header(
    "5. Recent Sensor Trends"
)


selected_sensor = st.selectbox(
    "Select sensor:",
    SENSOR_COLS
)


sensor_chart = (
    engine_df[
        ["cycle", selected_sensor]
    ]
    .tail(SEQUENCE_LENGTH)
    .set_index("cycle")
)


st.line_chart(
    sensor_chart
)


# ============================================================
# RECENT ENGINE DATA
# ============================================================

with st.expander(
    f"View latest {SEQUENCE_LENGTH} cycles"
):

    display_columns = [
        "unit",
        "cycle"
    ] + SENSOR_COLS

    st.dataframe(
        engine_df[
            display_columns
        ].tail(
            SEQUENCE_LENGTH
        ),
        use_container_width=True
    )


# ============================================================
# 6. SHAP EXPLANATION
# ============================================================

st.header(
    "6. Why This Prediction?"
)


if shap_df is not None:

    st.write(
        f"SHAP explanation loaded from: "
        f"`{shap_path.name}`"
    )


    # --------------------------------------------------------
    # GLOBAL SHAP FEATURE IMPORTANCE
    # --------------------------------------------------------

    if (
        "feature" in shap_df.columns
        and "mean_abs_shap" in shap_df.columns
    ):

        top_features = (
            shap_df[
                ["feature", "mean_abs_shap"]
            ]
            .copy()
            .sort_values(
                "mean_abs_shap",
                ascending=False
            )
            .head(10)
        )


        st.write(
            """
            The following features had the largest
            overall SHAP influence on the trained GRU model.
            """
        )


        st.bar_chart(
            top_features.set_index(
                "feature"
            )["mean_abs_shap"]
        )


        st.dataframe(
            top_features,
            use_container_width=True
        )


    # --------------------------------------------------------
    # LOCAL SHAP FORMAT
    # --------------------------------------------------------

    elif "feature" in shap_df.columns:

        st.dataframe(
            shap_df,
            use_container_width=True
        )


    else:

        st.info(
            "SHAP file was found, but its columns do not "
            "match the expected feature-importance format."
        )

else:

    st.info(
        """
        SHAP feature importance is not available in the
        current model directory.

        The GRU prediction itself is working independently
        of the SHAP explanation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Jet Engine Predictive Maintenance | "
    "GRU-based Remaining Useful Life Prediction"
)