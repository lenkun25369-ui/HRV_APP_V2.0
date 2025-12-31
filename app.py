import os, json, tempfile, subprocess
import streamlit as st
import requests

from shock_rate import predict_shock

st.title("Shock Risk Based on HRV")

qp = st.experimental_get_query_params()
token_q = qp.get("token", [""])[0]
obs_q   = qp.get("obs", [""])[0]

token = st.text_input("Token", value=token_q, type="password")
obs_url = st.text_input("Observation URL", value=obs_q)

@st.cache_resource
def _check_models_exist():
    assert os.path.exists("models/model_focalloss.h5"), "Missing models/model_focalloss.h5"
    assert os.path.exists("models/xgb_model.json"), "Missing models/xgb_model.json"
_check_models_exist()

def fetch_observation(token, obs_url):
    r = requests.get(obs_url, headers={"Authorization": f"Bearer {token}"}, verify=False, timeout=20)
    r.raise_for_status()
    return r.json()

# ✅ 自動執行
if token and obs_url:
    with st.spinner("Fetching Patient Data..."):
        obs = fetch_observation(token, obs_url)

    with st.expander("Patient Data (Click to Expand)", expanded=False):
        st.json(obs)

    # 用 /tmp 的暫存檔串流程
    with tempfile.TemporaryDirectory() as td:
        obs_path = os.path.join(td, "obs.json")
        ecg_csv  = os.path.join(td, "ECG_5min.csv")
        h0_csv   = os.path.join(td, "h0.csv")

        with open(obs_path, "w") as f:
            json.dump(obs, f)

        with st.spinner("Parsing ECG..."):
            subprocess.check_call(["python", "parse_fhir_ecg_to_csv.py", obs_path, ecg_csv])

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
                raise RuntimeError("generate_HRV_10_features.py failed")

        with st.spinner("Predicting shock risk..."):
            preds = predict_shock(h0_csv)  # numpy array

    st.success("Done")
    st.metric("Estimate Risk", f"{preds[0]*100:.2f}%")
else:
    st.info("Please enter Token and Observation URL to start calculation")
