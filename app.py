import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

from streamlit_autorefresh import st_autorefresh

# Re‑execute this script every 1 second (1000 ms):
st_autorefresh(interval=5000, limit=None, key="datarefresher")


# === Load Models & Encoders ===
A_DIR = "model_a_residual_life"
B_DIR = "model_b_chemical_stability"
res_model = joblib.load(os.path.join(A_DIR, "residual_life_model.lgb"))
res_enc = joblib.load(os.path.join(A_DIR, "chemistry_encoder.pkl"))
stab_model = joblib.load(os.path.join(B_DIR, "stability_classifier.lgb"))
stab_scaler = joblib.load(os.path.join(B_DIR, "feature_scaler.pkl"))
stab_enc_target = joblib.load(os.path.join(B_DIR, "stability_encoder.pkl"))

# === App Config ===
st.set_page_config(page_title="Unified Battery Dashboard", layout="wide")
st.markdown("""
<style>
.kpi-card {background:#f0f2f6; padding:10px; border-radius:8px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# === Centered Main Title ===
st.markdown("""
    <h1 style='text-align: center;'>🔋 SBX Battery Health Monitoring and Sorting Dashboard</h1>
""", unsafe_allow_html=True)

# === Sidebar Filters ===
st.sidebar.header("🔧 Controls & Filters")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
chemistry_filter = st.sidebar.multiselect("Chemistries", options=['Li-ion','LiFePO4','NMC 811'], default=['Li-ion','LiFePO4','NMC 811'])
min_cycle = st.sidebar.slider("Min Cycle Count", 0, 5000, 0)

if uploaded:
    # Read data once
    df = pd.read_csv(uploaded, parse_dates=['timestamp'])
    # Dynamic view selector
    views = ['Overview'] + sorted(df['battery_id'].unique())
    view = st.sidebar.selectbox("Select View", options=views, key='view')

    # Feature engineering
    df['voltage_per_current'] = df['voltage_drop_V'] / (df['current_draw_A'] + 1e-6)
    df['temp_delta'] = df['temp_end_C'] - df['temp_start_C']
    df['chemistry_encoded'] = res_enc.transform(df['chemistry'])
    df['is_charging'] = df['charge_duration_h'] > 1.0

    # Predictions
    res_feats = [
        'cycle_count','voltage_drop_V','current_draw_A',
        'impedance_1Hz_ohm','impedance_10Hz_ohm','impedance_100Hz_ohm',
        'phase_1Hz_deg','phase_10Hz_deg','temp_start_C','temp_end_C',
        'internal_resistance_ohm','total_run_hours','voltage_per_current','temp_delta','chemistry_encoded'
    ]
    df['predicted_residual_life'] = res_model.predict(df[res_feats])

    stab_feats = [
        'cycle_count','voltage_drop_V','current_draw_A',
        'impedance_1Hz_ohm','impedance_10Hz_ohm','impedance_100Hz_ohm',
        'phase_1Hz_deg','phase_10Hz_deg','temp_start_C','temp_end_C',
        'internal_resistance_ohm','total_run_hours','voltage_per_current','temp_delta'
    ]
    Xb = np.hstack([stab_scaler.transform(df[stab_feats]), df[['chemistry_encoded']].values])
    df['stability_prob'] = stab_model.predict_proba(Xb)[:,1]
    df['predicted_stability'] = stab_enc_target.inverse_transform((df['stability_prob']>0.5).astype(int))

    # Apply filters
    df = df[df['chemistry'].isin(chemistry_filter) & (df['cycle_count'] >= min_cycle)]

    if st.session_state.view == 'Overview':
        # ---------------------- OVERVIEW ----------------------
        st.subheader("📊 Fleet Overview")
        total_batt = df['battery_id'].nunique()
        pct_unstable = (df['predicted_stability']=='unstable').mean()*100
        avg_life = df['predicted_residual_life'].mean()
        charging = df[df['is_charging']==True]['battery_id'].nunique()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Batteries", total_batt)
        c2.metric("% Unstable", f"{pct_unstable:.1f}%")
        c3.metric("Avg Residual Life", f"{avg_life:.0f} cycles")
        c4.metric("Charging Now", charging)
        st.markdown("---")
        # Daily aggregations
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date').agg({
            'current_draw_A':'mean',
            'voltage_drop_V':'mean',
            'energy_wh':'sum',
            'battery_id':'count'
        }).reset_index().rename(columns={'battery_id':'cycles'})
        o1,o2 = st.columns(2)
        o1.plotly_chart(px.line(daily, x='date', y='current_draw_A', title='Avg Daily Current'), use_container_width=True)
        o2.plotly_chart(px.line(daily, x='date', y='voltage_drop_V', title='Avg Daily Voltage'), use_container_width=True)
        o3,o4 = st.columns(2)
        o3.plotly_chart(px.line(daily, x='date', y='energy_wh', title='Daily Energy (Wh)'), use_container_width=True)
        o4.plotly_chart(px.bar(daily, x='date', y='cycles', title='Cycles per Day'), use_container_width=True)
        st.markdown("---")
        st.subheader("⏱️ Charge Duration Distribution")
        st.plotly_chart(px.histogram(df, x='charge_duration_h', nbins=30), use_container_width=True)
        st.markdown("---")
        st.subheader("📈 Source Variation by Chemistry")
        comp = df.groupby('chemistry').agg({
            'voltage_per_current':'mean', 'impedance_1Hz_ohm':'mean'
        }).reset_index()
        st.plotly_chart(px.bar(comp, x='chemistry', y=['voltage_per_current','impedance_1Hz_ohm'], barmode='group'), use_container_width=True)

    else:
        # --------------------- DRILL-DOWN ---------------------
        bid = st.session_state.view
        d = df[df['battery_id'] == bid]
        st.subheader(f"🔋 Battery {bid} Detail")
        sd = d.iloc[-1]
        # KPI cards
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Predicted Residual Life", f"{sd['predicted_residual_life']:.0f} cycles")
        # classification
        if sd['predicted_residual_life'] >= 600:
            life_label = 'Healthy'
        elif sd['predicted_residual_life'] >= 300:
            life_label = 'Moderate'
        else:
            life_label = 'Critical'
        k2.metric("Life Classification", life_label)
        k3.metric("Stability Confidence", f"{sd['stability_prob']:.1%}")
        k4.metric("Latest Impedance 1Hz", f"{sd['impedance_1Hz_ohm']:.3f} Ω")
        # badge
        color_map = {'Healthy':'green','Moderate':'orange','Critical':'red'}
        st.markdown(
            f"<div style='text-align:center; margin:10px;'><span style='background:{color_map[life_label]}; color:white; padding:0.5em 1em; border-radius:10px; font-size:1.2em;'>{life_label}</span></div>",
            unsafe_allow_html=True
        )
        # alerts
        st.subheader("⚠️ Alert Panel")
        if sd['predicted_residual_life'] < 200:
            st.warning("Residual life below 200 cycles. Maintenance Required!")
        if sd['impedance_1Hz_ohm'] > 0.2:
            st.error("High Impedance Detected")
        if sd.isnull().any():
            st.info("Missing Data Present")
        st.markdown("---")
        # time-series
        t1, t2 = st.columns(2)
        t1.plotly_chart(px.line(d, x='timestamp', y=['voltage_drop_V','current_draw_A'], title='Voltage & Current'), use_container_width=True)
        t2.plotly_chart(px.line(d, x='timestamp', y=['temp_delta','internal_resistance_ohm'], title='Temp Δ & Resistance'), use_container_width=True)
        # charge duration
        st.subheader('⏳ Charge Duration History')
        st.plotly_chart(px.scatter(d, x='timestamp', y='charge_duration_h', title='Duration per Cycle'), use_container_width=True)
        st.markdown('---')
        st.subheader('📋 Cycle History')
        st.dataframe(d.reset_index(drop=True))
else:
    st.info("Upload a CSV file to begin.")
