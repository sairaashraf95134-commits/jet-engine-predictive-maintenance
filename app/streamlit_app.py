import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

from agent.graph import build_agent_graph


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AeroHealth AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"
DEEP_LEARNING_DIR = MODELS_DIR / "deep_learning"

GRU_MODEL_PATH = DEEP_LEARNING_DIR / "gru_model.keras"


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 20
RUL_REFERENCE = 125

SENSOR_COLS = [
    f"sensor_{i}"
    for i in range(1, 22)
]

NASA_COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + SENSOR_COLS
)


# ============================================================
# HELPER: FIND FILE
# ============================================================

def find_file(filename):

    matches = list(PROJECT_ROOT.rglob(filename))

    for path in matches:

        if path.is_file():
            return path

    return None


# ============================================================
# LOAD GRU MODEL
# ============================================================

@st.cache_resource
def load_gru_model():

    if not GRU_MODEL_PATH.exists():
        return None

    try:

        return load_model(
            GRU_MODEL_PATH,
            compile=False
        )

    except Exception:

        return None


gru_model = load_gru_model()


# ============================================================
# LOAD LANGGRAPH AGENT
# ============================================================

@st.cache_resource
def load_agent():

    try:

        return build_agent_graph()

    except Exception as e:

        return e


agent = load_agent()


# ============================================================
# LOAD SHAP DATA
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

                df = pd.read_csv(path)

                return df, path

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
        "train_FD001",
    ]

    for filename in possible_names:

        path = find_file(filename)

        if path is not None:
            return path

    return None


TRAIN_PATH = find_training_data()


# ============================================================
# READ TRAINING DATA
# ============================================================

@st.cache_data
def load_training_dataframe(train_path_string):

    if train_path_string is None:
        return None

    try:

        path = Path(train_path_string)

        # ----------------------------------------------------
        # First attempt: CSV with headers
        # ----------------------------------------------------

        df = pd.read_csv(path)

        if all(
            sensor in df.columns
            for sensor in SENSOR_COLS
        ):

            return df

        # ----------------------------------------------------
        # NASA TXT / whitespace format
        # ----------------------------------------------------

        df = pd.read_csv(
            path,
            sep=r"\s+|,",
            engine="python",
            header=None,
        )

        if df.shape[1] < 26:
            return None

        df = df.iloc[:, :26].copy()

        df.columns = NASA_COLUMNS

        return df

    except Exception:

        return None


training_df = load_training_dataframe(
    str(TRAIN_PATH)
    if TRAIN_PATH is not None
    else None
)


# ============================================================
# TRAINING SCALER
# ============================================================

@st.cache_resource
def create_training_scaler(train_path_string):

    if train_path_string is None:
        return None

    train_df = load_training_dataframe(
        train_path_string
    )

    if train_df is None:
        return None

    try:

        sensor_data = (
            train_df[SENSOR_COLS]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
        )

        sensor_data = sensor_data.dropna(
            how="all"
        )

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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ✈️ AeroHealth AI")

    st.caption(
        "Aircraft Health Intelligence"
    )

    st.divider()

    st.markdown("### 🧭 System")

    st.markdown(
        """
        **Purpose**

        Estimate aircraft engine Remaining Useful Life.

        **Dataset**

        NASA C-MAPSS FD001

        **Primary Model**

        🧠 GRU Deep Learning

        **Sequence**

        20 cycles × 21 sensors

        **AI Layer**

        🤖 LangGraph

        **Explainability**

        🔎 SHAP
        """
    )

    st.divider()

    st.markdown("### 🏆 Model Performance")

    st.metric(
        "GRU MAE",
        "9.51 cycles"
    )

    st.metric(
        "GRU RMSE",
        "13.62 cycles"
    )

    st.caption(
        "GRU achieved the best validation performance "
        "among the evaluated models."
    )

    st.divider()

    st.markdown("### 👩‍💻 Developer")

    st.markdown(
        """
        **Saira Ashraf**

        AeroHealth AI — Predictive Maintenance Prototype

        *Research, academic and decision-support use.*
        """
    )


# ============================================================
# MODEL VALIDATION
# ============================================================

if gru_model is None:

    st.error(
        "❌ GRU model could not be loaded."
    )

    st.code(
        str(GRU_MODEL_PATH)
    )

    st.stop()


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "✈️ AeroHealth AI"
)

st.caption(
    "AI-powered aircraft health prediction using "
    "Deep Learning, sensor analytics and Agentic AI."
)

st.markdown(
    "**GRU-based Remaining Useful Life Prediction** "
    "followed by LangGraph-powered interpretation, "
    "risk analysis and maintenance recommendation."
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.subheader("⚙️ System Status")

status1, status2, status3 = st.columns(3)

with status1:

    st.success(
        "🧠 GRU Model Ready"
    )

with status2:

    if training_scaler is not None:

        st.success(
            "📊 Training Scaler Ready"
        )

    else:

        st.error(
            "📊 Training Scaler Missing"
        )

with status3:

    if isinstance(agent, Exception):

        st.error(
            "🤖 LangGraph Agent Error"
        )

    else:

        st.success(
            "🤖 LangGraph Agent Ready"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "🔎 Model & Preprocessing Information"
):

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Primary Model**")
        st.write("GRU — Deep Learning")

        st.write("**Model Path**")

        st.code(
            str(GRU_MODEL_PATH)
        )

        st.write("**Input Shape**")

        st.code(
            str(gru_model.input_shape)
        )

        st.write("**Output Shape**")

        st.code(
            str(gru_model.output_shape)
        )

    with col2:

        st.write("**Sequence Length**")
        st.write("20 cycles")

        st.write("**Sensor Features**")
        st.write("21 sensors")

        st.write("**Dataset**")
        st.write("NASA C-MAPSS FD001")

        st.write("**Agent Layer**")
        st.write("LangGraph")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🏆 Model Performance")

st.caption(
    "Validation performance from the final model comparison. "
    "Lower MAE and RMSE indicate better performance."
)


comparison_df = pd.DataFrame(
    {
        "Model": [
            "GRU",
            "LSTM",
            "MLP",
            "Random Forest",
        ],

        "MAE": [
            9.5145,
            9.9230,
            11.66,
            13.6210,
        ],

        "RMSE": [
            13.6195,
            14.20,
            15.90,
            18.7999,
        ],
    }
)


display_comparison = comparison_df.copy()

display_comparison["MAE"] = (
    display_comparison["MAE"]
    .map(
        lambda x: f"{x:.2f}"
    )
)

display_comparison["RMSE"] = (
    display_comparison["RMSE"]
    .map(
        lambda x: f"{x:.2f}"
    )
)


st.dataframe(
    display_comparison,
    use_container_width=True,
    hide_index=True,
)


st.success(
    "🏆 GRU achieved the best validation performance "
    "among the evaluated models."
)


# ============================================================
# SYSTEM CAPABILITIES
# ============================================================

st.subheader("🛠️ System Capabilities")

cap1, cap2, cap3, cap4 = st.columns(4)

with cap1:

    st.markdown("### 🧠 Deep Learning")

    st.write(
        "GRU learns temporal degradation patterns "
        "from recent engine sensor sequences."
    )

with cap2:

    st.markdown("### 📈 Sensor Analytics")

    st.write(
        "Tracks recent sensor behaviour and "
        "degradation trends."
    )

with cap3:

    st.markdown("### 🔎 Explainability")

    st.write(
        "SHAP identifies features with the "
        "greatest model influence."
    )

with cap4:

    st.markdown("### 🤖 Agentic AI")

    st.write(
        "LangGraph interprets the prediction "
        "and generates risk-aware recommendations."
    )


# ============================================================
# UPLOAD
# ============================================================

st.subheader(
    "📂 Upload Engine Sensor Data"
)

uploaded_file = st.file_uploader(
    "Upload the engine sensor CSV",
    type=["csv"],
)


if uploaded_file is None:

    st.info(
        "Upload an engine sensor CSV to begin the analysis."
    )

    st.stop()


# ============================================================
# READ UPLOADED DATA
# ============================================================

try:

    uploaded_file.seek(0)

    test_df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read the uploaded CSV: {e}"
    )

    st.stop()


test_df.columns = (
    test_df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "unit",
    "cycle",
    *SENSOR_COLS,
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
# CLEAN DATA
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


test_df = (
    test_df
    .dropna(
        subset=[
            "unit",
            "cycle"
        ]
    )
    .sort_values(
        [
            "unit",
            "cycle"
        ]
    )
    .reset_index(drop=True)
)


if test_df.empty:

    st.error(
        "The uploaded dataset contains no valid records."
    )

    st.stop()


st.success(
    f"Successfully loaded {uploaded_file.name}"
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader(
    "📊 Dataset Overview"
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


c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Engines",
        total_engines
    )

with c2:

    st.metric(
        "Total Cycles",
        total_cycles
    )

with c3:

    st.metric(
        "Sensors",
        len(SENSOR_COLS)
    )

with c4:

    st.metric(
        "GRU Window",
        f"{SEQUENCE_LENGTH} cycles"
    )


with st.expander(
    "📋 View Uploaded Data"
):

    st.dataframe(
        test_df.head(30),
        use_container_width=True,
    )


# ============================================================
# ENGINE SELECTION
# ============================================================

st.subheader(
    "🔧 Engine Selection"
)

selected_engine = st.selectbox(
    "Select the engine to analyze",
    engine_ids,
)


engine_df = (
    test_df[
        test_df["unit"] == selected_engine
    ]
    .sort_values("cycle")
    .reset_index(drop=True)
)


total_engine_cycles = len(
    engine_df
)

latest_cycle = int(
    engine_df["cycle"].max()
)


if total_engine_cycles < SEQUENCE_LENGTH:

    st.error(
        f"Engine {int(selected_engine)} has only "
        f"{total_engine_cycles} cycles. "
        f"At least {SEQUENCE_LENGTH} cycles are required."
    )

    st.stop()


# ============================================================
# ENGINE METRICS
# ============================================================

e1, e2, e3 = st.columns(3)

with e1:

    st.metric(
        "Selected Engine",
        f"Engine {int(selected_engine)}"
    )

with e2:

    st.metric(
        "Latest Cycle",
        latest_cycle
    )

with e3:

    st.metric(
        "Cycles Available",
        total_engine_cycles
    )


# ============================================================
# PREPARE GRU INPUT
# ============================================================

raw_sequence_df = (
    engine_df[
        SENSOR_COLS
    ]
    .tail(
        SEQUENCE_LENGTH
    )
    .copy()
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

if raw_sequence_df.isna().sum().sum() > 0:

    st.warning(
        "Missing sensor values detected. "
        "Training-data medians will be used."
    )

    if training_df is not None:

        training_medians = (
            training_df[
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

    else:

        raw_sequence_df = (
            raw_sequence_df
            .fillna(0)
        )


# ============================================================
# SCALE
# ============================================================

if training_scaler is None:

    st.error(
        "Training StandardScaler is unavailable."
    )

    st.stop()


X_raw = raw_sequence_df.to_numpy(
    dtype=np.float32
)


X_scaled = training_scaler.transform(
    X_raw
).astype(np.float32)


X_input = X_scaled.reshape(
    1,
    SEQUENCE_LENGTH,
    len(SENSOR_COLS)
)


# ============================================================
# GRU INFERENCE
# ============================================================

try:

    prediction_output = gru_model.predict(
        X_input,
        verbose=0,
    )

    raw_prediction = float(
        np.asarray(
            prediction_output
        ).flatten()[0]
    )

except Exception as e:

    st.error(
        f"GRU prediction failed: {e}"
    )

    st.stop()


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
# GRU PREDICTION
# ============================================================

st.subheader(
    "🧠 GRU Prediction"
)

p1, p2, p3 = st.columns(3)

with p1:

    st.metric(
        "Remaining Useful Life",
        f"{predicted_rul:.2f} cycles"
    )

with p2:

    st.metric(
        "Health Status",
        health_status
    )

with p3:

    st.metric(
        "Prediction Model",
        "GRU"
    )


# ============================================================
# RUL INDICATOR
# ============================================================

st.subheader(
    "🎯 Remaining Useful Life Indicator"
)

indicator_value = min(
    max(
        predicted_rul / RUL_REFERENCE * 100,
        0
    ),
    100
)


st.progress(
    int(indicator_value)
)


st.caption(
    f"Estimated RUL: {predicted_rul:.2f} cycles "
    f"out of the {RUL_REFERENCE}-cycle reference scale."
)


if health_status == "Healthy":

    st.success(
        "The engine currently shows a relatively "
        "healthy predicted condition."
    )

elif health_status == "Degrading":

    st.warning(
        "The engine shows signs of degradation. "
        "Closer monitoring is recommended."
    )

else:

    st.error(
        "The predicted RUL is low. "
        "Engineering review and maintenance planning "
        "are recommended."
    )


# ============================================================
# RECENT SENSOR DATA
# ============================================================

recent_engine_window = (
    engine_df
    .tail(SEQUENCE_LENGTH)
    .copy()
)


recent_sensor_data = {}


for sensor in SENSOR_COLS:

    values = (
        recent_engine_window[sensor]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    if values.isna().all():

        values = pd.Series(
            [0.0] * len(values),
            index=values.index
        )

    else:

        values = values.fillna(
            values.median()
        )

    recent_sensor_data[sensor] = (
        values
        .astype(float)
        .tolist()
    )


# ============================================================
# SHAP FEATURE IMPORTANCE
# ============================================================

feature_importance = []


if shap_df is not None:

    if (
        "feature" in shap_df.columns
        and "mean_abs_shap" in shap_df.columns
    ):

        top_features = (
            shap_df[
                [
                    "feature",
                    "mean_abs_shap"
                ]
            ]
            .copy()
        )

        top_features["mean_abs_shap"] = (
            pd.to_numeric(
                top_features["mean_abs_shap"],
                errors="coerce"
            )
        )

        top_features = (
            top_features
            .dropna(
                subset=[
                    "mean_abs_shap"
                ]
            )
            .sort_values(
                "mean_abs_shap",
                ascending=False
            )
            .head(10)
        )

        for _, row in top_features.iterrows():

            feature_importance.append(
                {
                    "feature": str(
                        row["feature"]
                    ),

                    "importance": float(
                        row["mean_abs_shap"]
                    ),
                }
            )


# ============================================================
# RECENT SENSOR TRENDS
# ============================================================

st.divider()

st.subheader(
    "📈 Recent Sensor Trends"
)

selected_sensor = st.selectbox(
    "Select a sensor",
    SENSOR_COLS,
)


sensor_chart = (
    engine_df[
        [
            "cycle",
            selected_sensor
        ]
    ]
    .tail(SEQUENCE_LENGTH)
    .set_index("cycle")
)


st.line_chart(
    sensor_chart,
    use_container_width=True,
)


# ============================================================
# MULTIPLE SENSOR COMPARISON
# ============================================================

st.subheader(
    "📊 Compare Multiple Sensors"
)

selected_sensors = st.multiselect(
    "Select sensors to compare",
    SENSOR_COLS,

    default=[
        "sensor_2",
        "sensor_7",
        "sensor_11",
    ],
)


if selected_sensors:

    multi_sensor_chart = (
        engine_df[
            [
                "cycle",
                *selected_sensors
            ]
        ]
        .tail(SEQUENCE_LENGTH)
        .set_index("cycle")
    )

    st.line_chart(
        multi_sensor_chart,
        use_container_width=True,
    )


# ============================================================
# LATEST ENGINE WINDOW
# ============================================================

with st.expander(
    "📋 View Latest 20 Engine Cycles"
):

    st.dataframe(
        engine_df[
            [
                "unit",
                "cycle",
                *SENSOR_COLS
            ]
        ].tail(SEQUENCE_LENGTH),
        use_container_width=True,
    )


# ============================================================
# WHY THIS PREDICTION — SHAP
# ============================================================

st.subheader(
    "🔎 Why This Prediction?"
)

if shap_df is not None:

    st.caption(
        f"SHAP explanation loaded from: "
        f"{shap_path.name}"
    )

    if (
        "feature" in shap_df.columns
        and "mean_abs_shap" in shap_df.columns
    ):

        shap_display = (
            shap_df[
                [
                    "feature",
                    "mean_abs_shap"
                ]
            ]
            .copy()
        )

        shap_display["mean_abs_shap"] = (
            pd.to_numeric(
                shap_display["mean_abs_shap"],
                errors="coerce"
            )
        )

        shap_display = (
            shap_display
            .dropna(
                subset=[
                    "mean_abs_shap"
                ]
            )
            .sort_values(
                "mean_abs_shap",
                ascending=False
            )
            .head(10)
        )

        st.write(
            "Features with the largest overall influence "
            "on the trained GRU model:"
        )

        st.bar_chart(
            shap_display.set_index(
                "feature"
            )[
                "mean_abs_shap"
            ],
            use_container_width=True,
        )

        shap_table = shap_display.copy()

        shap_table["mean_abs_shap"] = (
            shap_table["mean_abs_shap"]
            .map(
                lambda x: f"{x:.4f}"
            )
        )

        st.dataframe(
            shap_table,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "The SHAP file does not contain the expected "
            "feature and mean_abs_shap columns."
        )

else:

    st.info(
        "SHAP feature importance is not available."
    )


# ============================================================
# SENSOR TREND ANALYSIS
#
# IMPORTANT:
# Percentage change is calculated between the first
# and last value of the latest 20-cycle window.
#
# Very small numerical changes are treated as noise.
# ============================================================

sensor_trends = []

for sensor in SENSOR_COLS:

    values = (
        recent_engine_window[sensor]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(values) < 2:
        continue

    first_value = float(
        values.iloc[0]
    )

    last_value = float(
        values.iloc[-1]
    )

    if abs(first_value) > 1e-9:

        percentage_change = (
            (
                last_value
                - first_value
            )
            / abs(first_value)
        ) * 100

    else:

        percentage_change = 0.0

    sensor_trends.append(
        {
            "sensor": sensor,
            "first": first_value,
            "latest": last_value,
            "change": percentage_change,
        }
    )


sensor_trends_df = pd.DataFrame(
    sensor_trends
)


# ============================================================
# LANGGRAPH AGENT INPUT
# ============================================================

agent_input = {

    "engine_id": int(
        selected_engine
    ),

    "latest_cycle": int(
        latest_cycle
    ),

    "predicted_rul": float(
        predicted_rul
    ),

    "recent_sensor_data":
        recent_sensor_data,

    "feature_importance":
        feature_importance,

    "user_question": "",
}


# ============================================================
# LANGGRAPH EXECUTION
# ============================================================

agent_result = {}


if isinstance(agent, Exception):

    st.error(
        f"LangGraph agent could not be created: {agent}"
    )

else:

    try:

        agent_result = agent.invoke(
            agent_input
        )

        if agent_result is None:

            agent_result = {}

    except Exception as e:

        st.error(
            f"LangGraph agent failed: {e}"
        )

        agent_result = {}


# ============================================================
# AGENT VALUES
# ============================================================

agent_status = agent_result.get(
    "health_status",
    health_status
)


agent_risk = agent_result.get(
    "risk_level",

    (
        "Low"
        if predicted_rul >= 70

        else "Moderate"
        if predicted_rul >= 40

        else "High"
        if predicted_rul >= 20

        else "Critical"
    )
)


validation_result = agent_result.get(
    "validation_result",
    "No validation result available."
)


assessment = agent_result.get(
    "final_assessment",

    (
        f"Engine {int(selected_engine)} has an estimated "
        f"Remaining Useful Life of {predicted_rul:.2f} cycles. "
        f"The model-based health status is {health_status}."
    )
)


sensor_analysis = agent_result.get(
    "sensor_analysis",
    "No sensor analysis available."
)


feature_analysis = agent_result.get(
    "feature_analysis",
    "No feature analysis available."
)


recommendation = agent_result.get(
    "recommendation",
    "No recommendation available."
)


# ============================================================
# AI AGENT REVIEW
# ============================================================

st.divider()

st.title(
    "🤖 AI Agent Review"
)

st.caption(
    "LangGraph-powered interpretation of the GRU prediction, "
    "sensor behaviour, model features and maintenance risk."
)


# ============================================================
# AGENT SUMMARY METRICS
# ============================================================

a1, a2, a3, a4 = st.columns(4)

with a1:

    st.metric(
        "Predicted RUL",
        f"{predicted_rul:.2f} cycles"
    )

with a2:

    st.metric(
        "Health Status",
        agent_status
    )

with a3:

    st.metric(
        "Risk Level",
        agent_risk
    )

with a4:

    st.metric(
        "Latest Cycle",
        latest_cycle
    )


# ============================================================
# OVERALL ASSESSMENT
# ============================================================

st.subheader(
    "🧠 Overall Assessment"
)

st.info(
    assessment
)


# ============================================================
# AGENT VALIDATION
# ============================================================

st.subheader(
    "✅ Agent Validation"
)

if "successful" in str(
    validation_result
).lower():

    st.success(
        validation_result
    )

else:

    st.warning(
        validation_result
    )


# ============================================================
# SENSOR BEHAVIOUR
# ============================================================

st.subheader(
    "📈 Sensor Behaviour Analysis"
)

st.write(
    sensor_analysis
)


# ============================================================
# SENSOR MOVEMENT SUMMARY
#
# IMPORTANT FIX:
# We do not present every tiny numerical difference
# as meaningful degradation.
# ============================================================

st.subheader(
    "📊 Sensor Movement Summary"
)


if not sensor_trends_df.empty:

    # --------------------------------------------------------
    # 0.05% threshold prevents meaningless 0.00% / 0.01%
    # changes from being presented as real degradation.
    # --------------------------------------------------------

    meaningful_threshold = 0.05

    increasing_df = (
        sensor_trends_df[
            sensor_trends_df["change"]
            >= meaningful_threshold
        ]
        .copy()
        .sort_values(
            "change",
            ascending=False
        )
    )

    decreasing_df = (
        sensor_trends_df[
            sensor_trends_df["change"]
            <= -meaningful_threshold
        ]
        .copy()
    )

    decreasing_df["abs_change"] = (
        decreasing_df["change"].abs()
    )

    decreasing_df = (
        decreasing_df
        .sort_values(
            "abs_change",
            ascending=False
        )
        .drop(
            columns=["abs_change"]
        )
    )


    trend_col1, trend_col2 = st.columns(2)


    # --------------------------------------------------------
    # INCREASING
    # --------------------------------------------------------

    with trend_col1:

        st.markdown(
            "### 📈 Increasing Sensors"
        )

        if increasing_df.empty:

            st.success(
                "No meaningful increasing sensor trend detected."
            )

        else:

            increase_display = (
                increasing_df[
                    [
                        "sensor",
                        "change"
                    ]
                ]
                .copy()
            )

            increase_display["change"] = (
                increase_display["change"]
                .map(
                    lambda x: f"+{x:.2f}%"
                )
            )

            st.dataframe(
                increase_display,
                use_container_width=True,
                hide_index=True,
            )


    # --------------------------------------------------------
    # DECREASING
    # --------------------------------------------------------

    with trend_col2:

        st.markdown(
            "### 📉 Decreasing Sensors"
        )

        if decreasing_df.empty:

            st.success(
                "No meaningful decreasing sensor trend detected."
            )

        else:

            decrease_display = (
                decreasing_df[
                    [
                        "sensor",
                        "change"
                    ]
                ]
                .copy()
            )

            decrease_display["change"] = (
                decrease_display["change"]
                .map(
                    lambda x: f"{x:.2f}%"
                )
            )

            st.dataframe(
                decrease_display,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# KEY RISK FACTORS
# ============================================================

st.subheader(
    "⚠️ Key Risk Factors"
)


if not sensor_trends_df.empty:

    meaningful_movement = (
        sensor_trends_df[
            sensor_trends_df["change"].abs()
            >= 0.05
        ]
        .copy()
        .sort_values(
            "change",
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    if meaningful_movement.empty:

        st.success(
            "No meaningful sensor movement was identified "
            "in the latest 20-cycle window."
        )

    else:

        for _, row in meaningful_movement.iterrows():

            sensor_name = row["sensor"]

            change = float(
                row["change"]
            )

            if change > 0:

                st.warning(
                    f"📈 {sensor_name} increased by "
                    f"{change:.2f}% over the recent "
                    f"20-cycle window."
                )

            else:

                st.info(
                    f"📉 {sensor_name} decreased by "
                    f"{abs(change):.2f}% over the recent "
                    f"20-cycle window."
                )


# ============================================================
# MODEL FEATURE ANALYSIS
# ============================================================

st.subheader(
    "🎯 Model Feature Analysis"
)


if feature_importance:

    feature_table = pd.DataFrame(
        feature_importance
    )

    feature_table = (
        feature_table
        .sort_values(
            "importance",
            ascending=False
        )
    )

    feature_display = feature_table.copy()

    feature_display["importance"] = (
        feature_display["importance"]
        .map(
            lambda x: f"{x:.4f}"
        )
    )

    st.dataframe(
        feature_display,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        feature_analysis
    )


# ============================================================
# RECOMMENDED ACTION
# ============================================================

st.subheader(
    "🛠️ Recommended Action"
)


if agent_status == "Healthy":

    st.success(
        recommendation
    )

elif agent_status == "Degrading":

    st.warning(
        recommendation
    )

else:

    st.error(
        recommendation
    )


# ============================================================
# FULL AGENT STATE
# ============================================================

with st.expander(
    "📋 View Full LangGraph Agent State"
):

    if agent_result:

        st.json(
            agent_result
        )

    else:

        st.info(
            "No LangGraph state is available."
        )


# ============================================================
# SYSTEM FLOW
# ============================================================

st.divider()

st.subheader(
    "🔄 AeroHealth AI System Flow"
)

st.caption(
    "From engine sensor data to intelligent maintenance insight"
)


flow1, flow2, flow3, flow4, flow5 = st.columns(5)


with flow1:

    st.markdown("### 📂")

    st.markdown(
        "**Engine Data**"
    )

    st.caption(
        "Sensor measurements"
    )


with flow2:

    st.markdown("### ⚙️")

    st.markdown(
        "**Preprocessing**"
    )

    st.caption(
        "Scaling + 20-cycle sequence"
    )


with flow3:

    st.markdown("### 🧠")

    st.markdown(
        "**GRU Model**"
    )

    st.caption(
        "RUL prediction"
    )


with flow4:

    st.markdown("### 🔎")

    st.markdown(
        "**Analytics**"
    )

    st.caption(
        "SHAP + sensor trends"
    )


with flow5:

    st.markdown("### 🤖")

    st.markdown(
        "**LangGraph**"
    )

    st.caption(
        "Risk + recommendation"
    )


# ============================================================
# IMPORTANT SYSTEM LIMITATION
# ============================================================

st.divider()

with st.expander(
    "⚠️ Important System Limitation"
):

    st.warning(
        """
        AeroHealth AI is a predictive analytics and
        decision-support prototype.

        GRU predictions and LangGraph interpretations
        should not be treated as certified aviation
        maintenance decisions.

        Actual aircraft maintenance decisions require
        appropriate engineering inspection, operational,
        regulatory and safety procedures.
        """
    )


# ============================================================
# FINAL BRAND MARK
# ============================================================

st.divider()

st.markdown(
    "### ✈️ 📊 🤖 AeroHealth AI"
)

st.caption(
    "AI-Powered Aircraft Health Prediction"
)

st.caption(
    "GRU Deep Learning • Sensor Analytics • LangGraph"
)

st.caption(
    "Developed by Saira Ashraf"
)