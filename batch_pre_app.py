import streamlit as st
import pandas as pd
import joblib
import os

# === Load Model and Encoder ===
model_dir = "model_a_residual_life"
model = joblib.load(os.path.join(model_dir, "residual_life_model.lgb"))
encoder = joblib.load(os.path.join(model_dir, "chemistry_encoder.pkl"))

# === Define Features ===
feature_names = [
    "cycle_count", "voltage_drop_V", "current_draw_A",
    "impedance_1Hz_ohm", "impedance_10Hz_ohm", "impedance_100Hz_ohm",
    "phase_1Hz_deg", "phase_10Hz_deg",
    "temp_start_C", "temp_end_C", "internal_resistance_ohm",
    "total_run_hours", "voltage_per_current", "temp_delta", "chemistry_encoded"
]

st.set_page_config(page_title="Battery Residual Life", layout="wide")
st.title("🔋 Battery Residual Life Prediction")
st.markdown("Use the sidebar to upload a CSV file for batch prediction.")

# === Sidebar Upload ===
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # === Check if 'chemistry' column exists and encode it ===
        if "chemistry" in df.columns:
            df["chemistry_encoded"] = encoder.transform(df["chemistry"])
        else:
            st.error("Column 'chemistry' is missing in the uploaded file.")
            st.stop()

        # === Feature Engineering ===
        df['voltage_per_current'] = df['voltage_drop_V'] / df['current_draw_A']
        df['temp_delta'] = df['temp_end_C'] - df['temp_start_C']

        # === Validate required columns ===
        missing = set(feature_names) - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        input_df = df[feature_names]
        predictions = model.predict(input_df)
        df['predicted_residual_life'] = predictions

        st.success("✅ Predictions completed!")
        st.dataframe(df[["cycle_count", "chemistry", "predicted_residual_life"]])

        # === Download ===
        csv = df.to_csv(index=False)
        st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
