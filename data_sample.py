import pandas as pd
import numpy as np

# Generate synthetic dataset with additional columns for unified dashboard
np.random.seed(123)

n_samples = 3000
battery_ids = [f"BATT_{i:03d}" for i in range(1, n_samples + 1)]
chemistries = ['Li-ion', 'LiFePO4', 'NMC 811']

# Sequential timestamps from Jan 1, 2023 to Dec 31, 2025
timestamps = pd.date_range(start='2023-01-01', end='2025-12-31', periods=n_samples)

# Generate basic features
cycle_count = np.random.randint(100, 1000, n_samples)
voltage_drop_V = np.round(np.random.uniform(0.05, 0.5, n_samples), 3)
current_draw_A = np.round(np.random.uniform(0.5, 5.0, n_samples), 3)
impedance_1Hz = np.round(np.random.uniform(0.1, 2.0, n_samples), 3)
impedance_10Hz = np.round(np.random.uniform(0.05, 1.0, n_samples), 3)
impedance_100Hz = np.round(np.random.uniform(0.01, 0.5, n_samples), 3)
phase_1Hz = np.round(np.random.uniform(-90, 90, n_samples), 1)
phase_10Hz = np.round(np.random.uniform(-90, 90, n_samples), 1)
temp_start_C = np.round(np.random.uniform(20, 30, n_samples), 1)
temp_end_C = np.round(np.random.uniform(20, 40, n_samples), 1)

# Derived features
internal_resistance = np.round((impedance_1Hz + impedance_10Hz + impedance_100Hz) / 3, 3)
voltage_per_current = np.round(voltage_drop_V / (current_draw_A + 1e-6), 3)
temp_delta = np.round(temp_end_C - temp_start_C, 1)
total_run_hours = cycle_count + np.random.randint(0, 20, n_samples)
remaining_cycles = np.round(np.maximum(0, 1000 - cycle_count + np.random.normal(0, 20, n_samples))).astype(int)
stability_status = np.where((internal_resistance > 0.6) | (impedance_1Hz > 1.5), 'unstable', 'stable')

# New columns for dashboard deliverables
# New columns
charge_duration = np.round(np.random.uniform(0.5, 5.0, n_samples), 2)
energy_wh = np.round(voltage_drop_V * current_draw_A * charge_duration, 2)
is_charging = charge_duration > 1.0

# Build DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'battery_id': battery_ids,
    'chemistry': np.random.choice(chemistries, n_samples),
    'cycle_count': cycle_count,
    'voltage_drop_V': voltage_drop_V,
    'current_draw_A': current_draw_A,
    'impedance_1Hz_ohm': impedance_1Hz,
    'impedance_10Hz_ohm': impedance_10Hz,
    'impedance_100Hz_ohm': impedance_100Hz,
    'phase_1Hz_deg': phase_1Hz,
    'phase_10Hz_deg': phase_10Hz,
    'temp_start_C': temp_start_C,
    'temp_end_C': temp_end_C,
    'internal_resistance_ohm': internal_resistance,
    'total_run_hours': total_run_hours,
    'voltage_per_current': voltage_per_current,
    'temp_delta': temp_delta,
    'charge_duration_h': charge_duration,
    'energy_wh': energy_wh,
    'is_charging': is_charging,
    'stability_status': stability_status,
    'remaining_cycles': remaining_cycles
})

# Display sample and save
file_path = 'extended_battery_data.csv'
df.to_csv(file_path, index=False)
file_path
