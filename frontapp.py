import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

# ───────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(page_title="Flight Price Prediction", page_icon="✈️", layout="wide")

st.title("✈️ Flight Price Prediction Dashboard")
st.markdown("##### Decision Tree & Random Forest — Regression and Classification on Flight Data")
st.markdown("---")


# ───────────────────────────────────────────────
# LOAD DATA
# ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Clean_Dataset.csv")
    drop_cols = [c for c in ["Unnamed: 0", "flight"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.drop_duplicates()

    def price_category(price):
        if price <= 10000:
            return "Budget"
        elif price <= 30000:
            return "Mid"
        else:
            return "Premium"

    df["price_category"] = df["price"].apply(price_category)
    return df


# ───────────────────────────────────────────────
# TRAIN ALL 4 MODELS
# ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Training all 4 models...")
def train_all_models(_df):
    df = _df.copy()

    cat_cols = ["airline", "source_city", "departure_time", "stops",
                "arrival_time", "destination_city", "class"]

    encoders = {}
    df_enc = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    # ── Regressor data ──
    X_reg = df_enc.drop(["price", "price_category"], axis=1)
    Y_reg = df_enc["price"]
    Xr_train, Xr_test, Yr_train, Yr_test = train_test_split(
        X_reg, Y_reg, test_size=0.2, random_state=42
    )

    # ── Classifier data ──
    X_clf = df_enc.drop(["price", "price_category"], axis=1)
    Y_clf = df_enc["price_category"]
    Xc_train, Xc_test, Yc_train, Yc_test = train_test_split(
        X_clf, Y_clf, test_size=0.2, random_state=42
    )

    common_params = dict(max_depth=15, min_samples_split=20, min_samples_leaf=10, random_state=42)

    # Decision Tree Regressor
    dt_reg = DecisionTreeRegressor(**common_params)
    dt_reg.fit(Xr_train, Yr_train)
    dt_reg_pred = dt_reg.predict(Xr_test)

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, **common_params)
    rf_reg.fit(Xr_train, Yr_train)
    rf_reg_pred = rf_reg.predict(Xr_test)

    # Decision Tree Classifier
    dt_clf = DecisionTreeClassifier(**common_params)
    dt_clf.fit(Xc_train, Yc_train)
    dt_clf_pred = dt_clf.predict(Xc_test)

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, **common_params)
    rf_clf.fit(Xc_train, Yc_train)
    rf_clf_pred = rf_clf.predict(Xc_test)

    scores = {
        "dt_reg_r2": r2_score(Yr_test, dt_reg_pred),
        "dt_reg_mae": mean_absolute_error(Yr_test, dt_reg_pred),
        "dt_reg_rmse": np.sqrt(mean_squared_error(Yr_test, dt_reg_pred)),
        "rf_reg_r2": r2_score(Yr_test, rf_reg_pred),
        "rf_reg_mae": mean_absolute_error(Yr_test, rf_reg_pred),
        "rf_reg_rmse": np.sqrt(mean_squared_error(Yr_test, rf_reg_pred)),
        "dt_clf_acc": accuracy_score(Yc_test, dt_clf_pred),
        "rf_clf_acc": accuracy_score(Yc_test, rf_clf_pred),
    }

    models = {
        "dt_reg": dt_reg, "rf_reg": rf_reg,
        "dt_clf": dt_clf, "rf_clf": rf_clf,
    }

    return models, encoders, scores, list(X_reg.columns)


df = load_data()
models, encoders, scores, feature_cols = train_all_models(df)


# ───────────────────────────────────────────────
# SIDEBAR — MODEL PICKER
# ───────────────────────────────────────────────
st.sidebar.header("⚙️ Choose Model")
model_choice = st.sidebar.radio(
    "Select algorithm + task",
    [
        "Decision Tree — Price (Regression)",
        "Random Forest — Price (Regression)",
        "Decision Tree — Price Category (Classification)",
        "Random Forest — Price Category (Classification)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Scores")
st.sidebar.markdown(f"**DT Regressor R²:** {scores['dt_reg_r2']:.2f}")
st.sidebar.markdown(f"**RF Regressor R²:** {scores['rf_reg_r2']:.2f}")
st.sidebar.markdown(f"**DT Classifier Acc:** {scores['dt_clf_acc']:.2f}")
st.sidebar.markdown(f"**RF Classifier Acc:** {scores['rf_clf_acc']:.2f}")


# ───────────────────────────────────────────────
# INPUT FORM (shared across all 4 models)
# ───────────────────────────────────────────────
st.subheader("✏️ Enter Flight Details")

col1, col2, col3 = st.columns(3)

with col1:
    airline = st.selectbox("Airline", sorted(df["airline"].unique()))
    source_city = st.selectbox("Source City", sorted(df["source_city"].unique()))
    destination_city = st.selectbox("Destination City", sorted(df["destination_city"].unique()))

with col2:
    departure_time = st.selectbox("Departure Time", sorted(df["departure_time"].unique()))
    arrival_time = st.selectbox("Arrival Time", sorted(df["arrival_time"].unique()))
    stops = st.selectbox("Stops", sorted(df["stops"].unique()))

with col3:
    travel_class = st.selectbox("Class", sorted(df["class"].unique()))
    duration = st.slider("Duration (hours)", float(df["duration"].min()), float(df["duration"].max()), 5.0)
    days_left = st.slider("Days Left Before Departure", int(df["days_left"].min()), int(df["days_left"].max()), 15)


def build_input_row():
    raw = {
        "airline": airline,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": travel_class,
        "duration": duration,
        "days_left": days_left,
    }
    row = {}
    for col in feature_cols:
        if col in encoders:
            row[col] = encoders[col].transform([raw[col]])[0]
        else:
            row[col] = raw[col]
    return pd.DataFrame([row])[feature_cols]


st.markdown("---")

if st.button("🔍 Predict", use_container_width=True):
    input_df = build_input_row()

    if model_choice == "Decision Tree — Price (Regression)":
        pred = models["dt_reg"].predict(input_df)[0]
        st.success(f"### 💰 Predicted Price: ₹{round(pred):,}")
        st.caption(f"Model: Decision Tree Regressor | R² Score: {scores['dt_reg_r2']:.2f} | RMSE: {scores['dt_reg_rmse']:.0f}")

    elif model_choice == "Random Forest — Price (Regression)":
        pred = models["rf_reg"].predict(input_df)[0]
        st.success(f"### 💰 Predicted Price: ₹{round(pred):,}")
        st.caption(f"Model: Random Forest Regressor | R² Score: {scores['rf_reg_r2']:.2f} | RMSE: {scores['rf_reg_rmse']:.0f}")

    elif model_choice == "Decision Tree — Price Category (Classification)":
        pred = models["dt_clf"].predict(input_df)[0]
        st.success(f"### 🏷️ Predicted Category: {pred}")
        st.caption(f"Model: Decision Tree Classifier | Accuracy: {scores['dt_clf_acc']:.2f}")

    else:
        pred = models["rf_clf"].predict(input_df)[0]
        st.success(f"### 🏷️ Predicted Category: {pred}")
        st.caption(f"Model: Random Forest Classifier | Accuracy: {scores['rf_clf_acc']:.2f}")

st.markdown("---")
st.caption("Dataset: Flight Price Prediction (Clean_Dataset.csv) | Models: Decision Tree & Random Forest")