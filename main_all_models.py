import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px

# === Load Models & Preprocessors ===
A_DIR = "model_a_residual_life"
B_DIR = "model_b_chemical_stability"
res_model = joblib.load(os.path.join(A_DIR, "residual_life_model.lgb"))
res_enc = joblib.load(os.path.join(A_DIR, "chemistry_encoder.pkl"))
stab_model = joblib.load(os.path.join(B_DIR, "stability_classifier.lgb"))
stab_enc_chem = joblib.load(os.path.join(B_DIR, "chemistry_encoder.pkl"))
stab_scaler = joblib.load(os.path.join(B_DIR, "feature_scaler.pkl"))
stab_enc_target = joblib.load(os.path.join(B_DIR, "stability_encoder.pkl"))

# === Constants ===
NUMERIC_STAB = [
    'cycle_count', 'voltage_drop_V', 'current_draw_A',
    'impedance_1Hz_ohm', 'impedance_10Hz_ohm', 'impedance_100Hz_ohm',
    'phase_1Hz_deg', 'phase_10Hz_deg', 'temp_start_C', 'temp_end_C',
    'internal_resistance_ohm', 'total_run_hours', 'voltage_per_current', 'temp_delta'
]
RES_FEATURES = NUMERIC_STAB + ['chemistry_encoded']

# === Page Setup ===
st.set_page_config(page_title="Battery Health Dashboard", layout="wide")
st.markdown("""
<h1 style='text-align: center;'>🔋 Unified Battery Health Monitoring Dashboard</h1>
""", unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.title("📂 Data Upload & Filters")
upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
chem_filter = st.sidebar.multiselect("Filter by Chemistry", options=['Li-ion','LiFePO4','NMC 811'], default=['Li-ion','LiFePO4','NMC 811'])
cycle_min = st.sidebar.slider("Min Cycle Count", 0, 5000, 0)

if upload:
    df = pd.read_csv(upload)

    # Validate
    expected_cols = ['battery_id','chemistry'] + NUMERIC_STAB
    missing = set(expected_cols) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # Common Feature Engineering
    df['voltage_per_current'] = df['voltage_drop_V'] / (df['current_draw_A'] + 1e-6)
    df['temp_delta'] = df['temp_end_C'] - df['temp_start_C']
    df['internal_resistance_ohm'] = df['internal_resistance_ohm'] if 'internal_resistance_ohm' in df else (
        (df['impedance_1Hz_ohm']+df['impedance_10Hz_ohm']+df['impedance_100Hz_ohm'])/3)

    # Residual Encoding & Prediction
    df['chemistry_encoded'] = res_enc.transform(df['chemistry'])
    df['predicted_residual_life'] = res_model.predict(df[RES_FEATURES])

    # Stability Encoding & Prediction
    df['chemistry_enc_b'] = stab_enc_chem.transform(df['chemistry'])
    Xb = np.hstack([stab_scaler.transform(df[NUMERIC_STAB]), df[['chemistry_enc_b']].values])
    df['stability_prob'] = stab_model.predict_proba(Xb)[:,1]
    df['predicted_stability'] = stab_enc_target.inverse_transform((df['stability_prob']>0.5).astype(int))

    # Filters
    df = df[df['chemistry'].isin(chem_filter) & (df['cycle_count']>=cycle_min)]

    # Fleet Summary
    total = len(df)
    critical_pct = (df['predicted_stability']=='unstable').mean()*100
    avg_life = df['predicted_residual_life'].mean()
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Batteries", total)
    c2.metric("% Unstable", f"{critical_pct:.1f}%")
    c3.metric("Avg Residual Life", f"{avg_life:.0f} cycles")

    # Select battery
    sel = st.selectbox("Select Battery ID", df['battery_id'].unique())
    row = df[df['battery_id']==sel].iloc[0]

    # Top KPIs
    r1,r2,r3,r4 = st.columns(4)
    r1.metric("Cycle Count", row['cycle_count'])
    r2.metric("Voltage Drop (V)", f"{row['voltage_drop_V']:.3f}")
    r3.metric("Internal Resistance (Ω)", f"{row['internal_resistance_ohm']:.3f}")
    r4.metric("Temp Δ (°C)", f"{row['temp_delta']:.1f}")

    # Gauges
    g1,g2 = st.columns(2)
    pct_life = row['predicted_residual_life']/1000
    g1.subheader("🔋 Residual Life Gauge")
    g1.progress(min(pct_life,1.0))
    g1.metric("Residual Life", f"{row['predicted_residual_life']:.0f} cycles")
    g2.subheader("🧪 Stability Gauge")
    g2.progress(row['stability_prob'])
    g2.metric("Stability", row['predicted_stability'], f"{row['stability_prob']:.1%}")

    # Trends Sparklines
    st.subheader("📈 Trends Overview")
    tcols = st.columns(3)
    for ix,col in enumerate(['predicted_residual_life','stability_prob','internal_resistance_ohm']):
        fig = px.line(df, x='battery_id', y=col, title=col.replace('_',' ').title())
        tcols[ix].plotly_chart(fig, use_container_width=True)

    # Alerts
    st.subheader("⚠️ Alerts")
    if row['predicted_residual_life']<200: st.warning("Low Residual Life (<200 cycles)")
    if row['stability_prob']<0.5: st.error("Unstable Chemistry Detected")

    # Batch Table & Download
    st.subheader("📋 Batch Data")
    st.dataframe(df[['battery_id','chemistry','predicted_residual_life','predicted_stability','stability_prob']])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "battery_health.csv")
else:
    st.info("Upload a CSV to run predictions.")
