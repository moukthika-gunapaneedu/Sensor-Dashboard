# Real-Time Sensor Dashboard (Simulated)

This Streamlit app simulates real-time sensor data for a pressurized container and
renders a live dashboard with:

- **Live values** for temperature & pressure
- **Multi-sensor redundancy** (2oo3 voting for temperature; 2 sensors for pressure)
- **Thresholds** (normal & SIS/emergency)
- **Anomaly detection** (z-score & rate-of-change)
- **Interlocks & Permissives**
- **SIS Trip** (latched shutdown if SIS threshold is breached)
- **Hoverable plots** with **day/week/month** time windows
- Adjustable **refresh rate**

## Quickstart

```bash
# 1) (optional) create & activate a venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

The app simulates one data point every refresh. Use the sidebar to tune thresholds,
toggle interlocks, and set the time window. SIS trips latch until reset.# Sensor-Dashboard
