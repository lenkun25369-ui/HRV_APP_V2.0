import os, json, tempfile, subprocess
import streamlit as st
import requests
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt

from shock_rate import predict_shock

# =========================================
# UI Header
# =========================================
st.title("SHIELD")
st.caption("HRV Sepsis Early Warning System Powerd by AI")

risk_placeholder = st.empty()

qp = st.experimental_get_query_params()
token_q = qp.get("token", [""])[0]
obs_q   = qp.get("obs", [""])[0]

# =========================================
# Check Models
# =========================================
@st.cache_resource
def _check_models_exist():
    assert os.path.exists("models/model_focalloss.h5"), "Missing models/model_focalloss.h5"
    assert os.path.exists("models/xgb_model.json"), "Missing models/xgb_model.json"

_check_models_exist()

# =========================================
# FHIR Fetch
# =========================================
def fetch_observation(token, obs_url):
    r = requests.get(
        obs_url,
        headers={"Authorization": f"Bearer {token}"},
        verify=False,
        timeout=20
    )
    r.raise_for_status()
    return r.json()

# =========================================
# Patient Data (Top)
# =========================================
patient_data_placeholder = st.empty()

with patient_data_placeholder.container():
    st.expander("Patient Data (Click to Expand)", expanded=False)

# =========================================
# Token & Observation URL (Bottom)
# =========================================
token = st.text_input("Token", value=token_q, type="password")
obs_url = st.text_input("Observation URL", value=obs_q)

# =========================================
# Auto Run Logic (UNCHANGED)
# =========================================
if token and obs_url:
    with st.spinner("Fetching Patient Data..."):
        obs = fetch_observation(token, obs_url)

    with patient_data_placeholder.container():
        with st.expander("Patient Data (Click to Expand)", expanded=False):
            st.json(obs)

    with tempfile.TemporaryDirectory() as td:
        obs_path = os.path.join(td, "obs.json")
        ecg_csv  = os.path.join(td, "ECG_5min.csv")
        h0_csv   = os.path.join(td, "h0.csv")

        with open(obs_path, "w") as f:
            json.dump(obs, f)

        with st.spinner("Parsing ECG..."):
            subprocess.check_call([
                "python",
                "parse_fhir_ecg_to_csv.py",
                obs_path,
                ecg_csv
            ])

        with st.spinner("Generating HRV features..."):
            proc = subprocess.run(
                [
                    "python",
                    "generate_HRV_10_features.py",
                    ecg_csv,
                    h0_csv
                ],
                capture_output=True,
                text=True
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr)

        with st.spinner("Predicting shock risk..."):
            preds = predict_shock(h0_csv)

    st.success("Done")

    with risk_placeholder.container():
        risk_pct = round(preds[0] * 100, 2)

    # =========================================
    # Risk Level
    # =========================================
    if risk_pct < 20:
        risk_label = "LOW RISK"
        risk_color = "#2ecc71"
    elif risk_pct < 40:
        risk_label = "MODERATE RISK"
        risk_color = "#f39c12"
    else:
        risk_label = "HIGH RISK"
        risk_color = "#e74c3c"

    # =========================================
    # Risk Visualization
    # =========================================
    with risk_placeholder.container():
        pie_col, value_col = st.columns([1, 2], gap="large")

        # ----- Pie -----
        with pie_col:
            components.html(
                f"""
                <style>
                .pie {{
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    background: conic-gradient(
                        {risk_color} {risk_pct}%,
                        #2c2c2c {risk_pct}% 100%
                    );
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .pie-inner {{
                    width: 70px;
                    height: 70px;
                    background: #0e1117;
                    border-radius: 50%;
                }}
                </style>
                <div style="display:flex; justify-content:center;">
                    <div class="pie">
                        <div class="pie-inner"></div>
                    </div>
                </div>
                """,
                height=140,
            )

        # ----- Value -----
        with value_col:
            st.markdown(
                f"""
                <div style="text-align:center; margin-top:18px;">
                    <div style="font-size:42px; font-weight:800;">
                        {risk_pct:.2f}%
                    </div>
                    <div style="font-size:20px; font-weight:700; color:{risk_color};">
                        {risk_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================
    # NEW: ECG Input & HRV Output
    # =========================================
    st.markdown("---")
    st.subheader("ECG Input & HRV Features")

    # ----- ECG Plot -----
    try:
        ecg_df = pd.read_csv(ecg_csv, header=None)
        ecg_signal = ecg_df.iloc[:, 0].values

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ecg_signal, linewidth=1)
        ax.set_title("ECG Signal (Input to HRV Generator)")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Failed to plot ECG signal: {e}")

    # ----- HRV Table -----
    try:
        hrv_df = pd.read_csv(h0_csv)
        st.markdown("**HRV Features Output**")
        st.dataframe(hrv_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Failed to load HRV features: {e}")

else:
    st.info("Please enter Token and Observation URL to start calculation")
