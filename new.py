import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# === Load Model and Encoder ===
model_dir = "model_a_residual_life"
model = joblib.load(os.path.join(model_dir, "residual_life_model.lgb"))
encoder = joblib.load(os.path.join(model_dir, "chemistry_encoder.pkl"))

# === Sidebar: Upload CSV and Filters ===
st.sidebar.title("📂 Data Upload & Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Placeholder for future filters
chem_filter = st.sidebar.selectbox("Filter by Chemistry", options=["All", "Li-ion", "NiMH", "NiCd"], index=0)
cycle_threshold = st.sidebar.slider("Min Cycle Count", 0, 2000, 100)

# === Initialize main content ===
st.title("🔋 Battery Residual Life Monitoring Dashboard")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Encode chemistry
    if "chemistry" in df.columns:
        df["chemistry_encoded"] = encoder.transform(df["chemistry"])
    else:
        st.error("Missing 'chemistry' column")
        st.stop()

    # Feature Engineering
    df['voltage_per_current'] = df['voltage_drop_V'] / df['current_draw_A']
    df['temp_delta'] = df['temp_end_C'] - df['temp_start_C']

    feature_names = [
        "cycle_count", "voltage_drop_V", "current_draw_A",
        "impedance_1Hz_ohm", "impedance_10Hz_ohm", "impedance_100Hz_ohm",
        "phase_1Hz_deg", "phase_10Hz_deg",
        "temp_start_C", "temp_end_C", "internal_resistance_ohm",
        "total_run_hours", "voltage_per_current", "temp_delta", "chemistry_encoded"
    ]

    missing = set(feature_names) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    # Predict
    df['predicted_residual_life'] = model.predict(df[feature_names])

    # Filter data
    if chem_filter != "All":
        df = df[df['chemistry'] == chem_filter]
    df = df[df['cycle_count'] >= cycle_threshold]

    selected_battery = st.sidebar.selectbox("Select Battery ID", df.index.astype(str))
    selected_data = df.loc[int(selected_battery)] if selected_battery else df.iloc[0]

    # === KPI Cards (Top Row) ===
    col1, col2, col3 = st.columns(3)
    col1.metric("Battery ID", selected_battery)
    col2.metric("Chemistry", selected_data['chemistry'])
    col3.metric("Predicted Residual Life", f"{selected_data['predicted_residual_life']:.2f} cycles")

    # === Status Gauge (Placeholder using color) ===
    life_pct = (selected_data['predicted_residual_life'] / 1000) * 100
    st.markdown(f"### 🔵 Battery Life Gauge: {life_pct:.1f}% Remaining")
    st.progress(life_pct / 100)

    # === Sparkline Trends ===
    st.subheader("📈 Trends Overview")
    trend_cols = st.columns(3)
    for i, col in enumerate(['voltage_drop_V', 'internal_resistance_ohm', 'predicted_residual_life']):
        fig = px.line(df, y=col, title=col.replace('_', ' ').title())
        trend_cols[i].plotly_chart(fig, use_container_width=True)

    # === KPI Cards (Second Row) ===
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Latest Impedance 1Hz", f"{selected_data['impedance_1Hz_ohm']:.3f} Ω")
    kpi2.metric("Temp Delta", f"{selected_data['temp_delta']:.2f} °C")
    kpi3.metric("Cycle Count", f"{selected_data['cycle_count']}")

    # === Alerts Panel ===
    st.subheader("⚠️ Alert Panel")
    if selected_data['predicted_residual_life'] < 200:
        st.warning("Residual life below 200 cycles. Maintenance Required!")
    if selected_data['impedance_1Hz_ohm'] > 0.2:
        st.error("High Impedance Detected")
    if pd.isnull(selected_data).any():
        st.info("Missing Data Present")

    # === Batch Table ===
    st.subheader("📋 Batch Predictions")
    st.dataframe(df[["cycle_count", "chemistry", "predicted_residual_life"]].sort_values(by="predicted_residual_life"))

    # === Download Button ===
    st.download_button("Download Results as CSV", df.to_csv(index=False), "predictions.csv")
else:
    st.info("Please upload a CSV file to begin.")
