\
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sensor Dashboard (Simulated)", layout="wide", page_icon="📈")

# ---------- Utilities ----------

def init_state():
    ss = st.session_state
    if "data" not in ss:
        ss.data = pd.DataFrame(columns=[
            "ts",
            "T1","T2","T3",   # three temp sensors
            "P1","P2",       # two pressure sensors
            "T_voted","P_voted",
            "T_anom","P_anom",
            "any_sensor_fault",
            "sis_tripped"
        ])
    if "sis_tripped" not in ss:
        ss.sis_tripped = False
    if "last_update" not in ss:
        ss.last_update = None
    if "seed" not in ss:
        ss.seed = int(time.time()) % 2_147_483_647
    if "base_T" not in ss:
        ss.base_T = 70.0  # deg C (starting baseline)
    if "base_P" not in ss:
        ss.base_P = 2.0   # bar (starting baseline)

def simulate_point(noise_T=0.6, noise_P=0.03, spike_prob=0.01, stuck_prob=0.003):
    """
    Simulate a new measurement for each sensor with some dynamics:
      - random walk around base
      - occasional spikes
      - occasional sensor stuck fault
    """
    rng = np.random.default_rng(st.session_state.seed + int(time.time()))
    ss = st.session_state

    # drift baseline slightly
    ss.base_T += rng.normal(0, 0.05)
    ss.base_P += rng.normal(0, 0.002)

    def sensor_value(base, noise, allow_spike=True, allow_stuck=True, key=None):
        val = base + rng.normal(0, noise)
        # occasional spike
        if allow_spike and rng.random() < spike_prob:
            val += rng.normal(5*noise, 10*noise)
        # occasional stuck: keep previous reading if exists
        if allow_stuck and rng.random() < stuck_prob and key is not None and len(ss.data) > 0:
            val = ss.data.iloc[-1][key]
        return val

    T1 = sensor_value(ss.base_T, noise_T, key="T1")
    T2 = sensor_value(ss.base_T, noise_T, key="T2")
    T3 = sensor_value(ss.base_T, noise_T, key="T3")
    P1 = sensor_value(ss.base_P, noise_P, key="P1")
    P2 = sensor_value(ss.base_P, noise_P, key="P2")

    # 2oo3 (median) voter for Temperature; median for Pressure (2 sensors)
    T_voted = np.median([T1, T2, T3])
    P_voted = np.median([P1, P2])

    # simple sensor fault flags: outlier vs voter by > 3*noise or no movement over 5 samples
    any_fault = False
    for key, val, voter, tol in [("T1",T1,T_voted,3*noise_T),("T2",T2,T_voted,3*noise_T),("T3",T3,T_voted,3*noise_T),
                                 ("P1",P1,P_voted,3*noise_P),("P2",P2,P_voted,3*noise_P)]:
        if abs(val - voter) > tol:
            any_fault = True

    # anomaly detection: z-score in sliding window + rate-of-change
    df = st.session_state.data
    now = datetime.utcnow()
    if len(df) >= 30:
        recent = df.tail(30)
        def z(x, mu, sd):
            return 0 if sd == 0 else (x - mu)/sd
        T_mu, T_sd = recent["T_voted"].mean(), recent["T_voted"].std()
        P_mu, P_sd = recent["P_voted"].mean(), recent["P_voted"].std()
        T_z = z(T_voted, T_mu, T_sd)
        P_z = z(P_voted, P_mu, P_sd)
        dTdt = T_voted - recent["T_voted"].iloc[-1]
        dPdt = P_voted - recent["P_voted"].iloc[-1]
        T_anom = abs(T_z) > 3 or abs(dTdt) > 5.0
        P_anom = abs(P_z) > 3 or abs(dPdt) > 0.2
    else:
        T_anom = False
        P_anom = False

    return {
        "ts": now,
        "T1": float(T1), "T2": float(T2), "T3": float(T3),
        "P1": float(P1), "P2": float(P2),
        "T_voted": float(T_voted),
        "P_voted": float(P_voted),
        "T_anom": bool(T_anom),
        "P_anom": bool(P_anom),
        "any_sensor_fault": bool(any_fault),
        "sis_tripped": bool(st.session_state.sis_tripped),
    }

def within(val, low, high):
    return (val >= low) and (val <= high)

# ---------- UI ----------

init_state()
st.title("📈 Real-Time Sensor Dashboard (Simulated)")
st.caption("Container temperature & pressure with redundancy voting, thresholds, anomalies, interlocks, and SIS trip.")

with st.sidebar:
    st.header("⚙️ Controls")
    refresh_sec = st.slider("Refresh interval (seconds)", 1, 10, 2, help="How often to fetch & render new data.")
    win = st.selectbox("Time window", ["Day","Week","Month"], index=0)
    # For simulation, keep everything but we will filter by timedelta
    window_map = {"Day": timedelta(days=1), "Week": timedelta(weeks=1), "Month": timedelta(days=30)}

    st.subheader("Thresholds")
    # Normal operating thresholds
    T_low = st.number_input("Temp lower (°C)", value=60.0, step=0.5)
    T_high = st.number_input("Temp upper (°C)", value=90.0, step=0.5)
    P_low = st.number_input("Pressure lower (bar)", value=1.5, step=0.05, format="%.2f")
    P_high = st.number_input("Pressure upper (bar)", value=2.5, step=0.05, format="%.2f")

    st.subheader("SIS Emergency Thresholds (Trip)")
    T_sis_low  = st.number_input("SIS Temp low (°C)", value=55.0, step=0.5)
    T_sis_high = st.number_input("SIS Temp high (°C)", value=95.0, step=0.5)
    P_sis_low  = st.number_input("SIS Pressure low (bar)", value=1.2, step=0.05, format="%.2f")
    P_sis_high = st.number_input("SIS Pressure high (bar)", value=2.8, step=0.05, format="%.2f")

    st.subheader("Interlocks & Permissives")
    power_ok = st.checkbox("Power Available", value=True)
    valve_ok = st.checkbox("Valves Correctly Positioned", value=True)
    pump_ok  = st.checkbox("Pump Healthy", value=True)
    manual_permit = st.checkbox("Manual Permit Enabled", value=True)

    st.subheader("SIS Trip Control")
    if st.button("🔁 Reset SIS Trip"):
        st.session_state.sis_tripped = False

# generate one new data point per refresh
new_row = simulate_point()
df = st.session_state.data

# Check SIS trip logic (latched)
if (new_row["T_voted"] < T_sis_low or new_row["T_voted"] > T_sis_high or
    new_row["P_voted"] < P_sis_low or new_row["P_voted"] > P_sis_high):
    st.session_state.sis_tripped = True
new_row["sis_tripped"] = st.session_state.sis_tripped

# Append
st.session_state.data = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Filter by time window
now = datetime.utcnow()
window = window_map[win]
dfw = st.session_state.data[st.session_state.data["ts"] >= (now - window)].copy()

# Compute status
T_ok    = within(new_row["T_voted"], T_low, T_high)
P_ok    = within(new_row["P_voted"], P_low, P_high)
perm_ok = power_ok and valve_ok and pump_ok and manual_permit
overall_ok = T_ok and P_ok and (not new_row["any_sensor_fault"]) and perm_ok and (not st.session_state.sis_tripped)

# ---------- KPI Row ----------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Temp (°C) [voted]", f"{new_row['T_voted']:.2f}", help="Median of T1,T2,T3")
kpi2.metric("Pressure (bar) [voted]", f"{new_row['P_voted']:.3f}", help="Median of P1,P2")
kpi3.metric("Sensor fault?", "Yes" if new_row["any_sensor_fault"] else "No")
kpi4.metric("Anomaly (T/P)", f"{'T' if new_row['T_anom'] else '-'} / {'P' if new_row['P_anom'] else '-'}")
kpi5.metric("SIS Trip", "TRIPPED" if st.session_state.sis_tripped else "OK")

# ---------- Status & Interlocks ----------
st.subheader("System Status")
status_color = "#16a34a" if overall_ok else ("#eab308" if not st.session_state.sis_tripped else "#dc2626")
st.markdown(f"""
<div style="padding:10px;border-radius:8px;background:{status_color};color:white;">
<b>Overall:</b> {"OK & Permissives Satisfied" if overall_ok else ("WARNING (Check thresholds/sensors/interlocks)" if not st.session_state.sis_tripped else "SIS TRIPPED — SYSTEM SHUTDOWN")}
</div>
""", unsafe_allow_html=True)

cols = st.columns(4)
with cols[0]:
    st.write("**Permissives**")
    st.write(f"- Power: {'✅' if power_ok else '❌'}")
    st.write(f"- Valves: {'✅' if valve_ok else '❌'}")
    st.write(f"- Pump: {'✅' if pump_ok else '❌'}")
    st.write(f"- Manual Permit: {'✅' if manual_permit else '❌'}")
with cols[1]:
    st.write("**Threshold Checks**")
    st.write(f"- Temp within normal: {'✅' if T_ok else '❌'}")
    st.write(f"- Pressure within normal: {'✅' if P_ok else '❌'}")
with cols[2]:
    st.write("**Anomalies**")
    st.write(f"- Temp anomaly: {'⚠️' if new_row['T_anom'] else '—'}")
    st.write(f"- Pressure anomaly: {'⚠️' if new_row['P_anom'] else '—'}")
with cols[3]:
    st.write("**Voting**")
    st.write(f"Temp sensors: T1={new_row['T1']:.2f}, T2={new_row['T2']:.2f}, T3={new_row['T3']:.2f}")
    st.write(f"Pressure sensors: P1={new_row['P1']:.3f}, P2={new_row['P2']:.3f}")

# ---------- Plots ----------
def add_threshold_traces(fig, y_low, y_high, name_low, name_high, xref):
    fig.add_hline(y=y_low, line_dash="dot", line_width=1, annotation_text=name_low, annotation_position="top left")
    fig.add_hline(y=y_high, line_dash="dot", line_width=1, annotation_text=name_high, annotation_position="top left")

# Temperature
figT = go.Figure()
figT.add_trace(go.Scatter(x=dfw["ts"], y=dfw["T_voted"], name="Temp (voted)", mode="lines"))
figT.add_trace(go.Scatter(x=dfw["ts"], y=dfw["T1"], name="T1", mode="lines", opacity=0.35))
figT.add_trace(go.Scatter(x=dfw["ts"], y=dfw["T2"], name="T2", mode="lines", opacity=0.35))
figT.add_trace(go.Scatter(x=dfw["ts"], y=dfw["T3"], name="T3", mode="lines", opacity=0.35))
add_threshold_traces(figT, T_low, T_high, "Normal Low", "Normal High", "x")
add_threshold_traces(figT, T_sis_low, T_sis_high, "SIS Low", "SIS High", "x")
figT.update_layout(title="Temperature vs Time", xaxis_title="Time (UTC)", yaxis_title="°C", hovermode="x unified")

# Pressure
figP = go.Figure()
figP.add_trace(go.Scatter(x=dfw["ts"], y=dfw["P_voted"], name="Pressure (voted)", mode="lines"))
figP.add_trace(go.Scatter(x=dfw["ts"], y=dfw["P1"], name="P1", mode="lines", opacity=0.35))
figP.add_trace(go.Scatter(x=dfw["ts"], y=dfw["P2"], name="P2", mode="lines", opacity=0.35))
add_threshold_traces(figP, P_low, P_high, "Normal Low", "Normal High", "x")
add_threshold_traces(figP, P_sis_low, P_sis_high, "SIS Low", "SIS High", "x")
figP.update_layout(title="Pressure vs Time", xaxis_title="Time (UTC)", yaxis_title="bar", hovermode="x unified")

c1, c2 = st.columns(2)
with c1: st.plotly_chart(figT, use_container_width=True)
with c2: st.plotly_chart(figP, use_container_width=True)

# ---------- Table (latest N) ----------
st.subheader("Latest Samples")
st.dataframe(dfw.tail(25).set_index("ts"))

# ---------- Auto-refresh ----------
# Streamlit reruns the script top-to-bottom; sleep for refresh interval
time.sleep(refresh_sec)
st.experimental_rerun()
