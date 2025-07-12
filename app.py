
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and scaler
rf_model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- RESET FUNCTION ---
def reset_fields():
    st.session_state.air_temp = ""
    st.session_state.process_temp = ""
    st.session_state.rotation_speed = ""
    st.session_state.torque = ""
    st.session_state.tool_wear = ""
    st.session_state.reset_flag = True
    st.rerun()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sistem Ramalan Penyenggaraan Transit Aliran Ringan", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #6b2929;
        color: white;
    }

    h1, h2, h3, h4, h5, h6, label, p, span, div {
        color: white !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div,
    .stMultiSelect div,
    .stTextArea textarea {
    color: black !important;               
    background-color: white !important;   

    }

    .stDataFrame div,
    .stDataFrame table,
    .stDataFrame th,
    .stDataFrame td {
        color: black !important;
        background-color: white !important;
    }

    .stButton > button {
        background-color: #ffcccc;
        color: black;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
        width: 100%;
        border: none;
        transition: background-color 0.3s;
    }

    .stButton > button:hover {
        background-color: #e57373;
        color: white;
    }

    .input-box {
        background-color: #d19b96;
        padding: 10px;
        border-radius: 10px;
        width: 100%;
        margin-top: 6px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }

    .failure-box, .no-failure-box {
        color: white;
        padding: 14px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }

    .failure-box {
        background-color: #c62828;
        width: 100%;
    }

    .no-failure-box {
        background-color: #00c853;
        width: 100%;
    }

    table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 5px;
        font-size: 14px;
    }

    td {
        background-color: white;
        padding: 8px;
        border-radius: 5px;
        text-align: center;
        color: black;
    }

    th {
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- INIT STATE ---
for key in ["air_temp", "process_temp", "rotation_speed", "torque", "tool_wear", "reset_flag", "previous_data"]:
    if key not in st.session_state:
        if key == "previous_data":
            st.session_state[key] = []
        elif key == "reset_flag":
            st.session_state[key] = False
        else:
            st.session_state[key] = ""

# --- TITLE ---
st.markdown("<h1 style='text-align: center;'>PREDICTIVE MAINTENANCE OF LIGHT RAIL TRANSIT (LRT)</h1>", unsafe_allow_html=True)

# --- LAYOUT ---
col1, col2 = st.columns([1, 2])

# --- INPUT FORM (LEFT) ---
with col1:
    st.markdown("""<div class='input-box'><h3>Enter Data</h3>""", unsafe_allow_html=True)

    st.text_input("Air temperature [K]:", key="air_temp")
    st.text_input("Process temperature [K]:", key="process_temp")
    st.text_input("Rotational speed [rpm]:", key="rotation_speed")
    st.text_input("Torque [Nm]:", key="torque")
    st.text_input("Tool wear [min]:", key="tool_wear")

    col_submit, col_reset = st.columns(2)
    with col_submit:
        predict_btn = st.button("SUBMIT")
    with col_reset:
        reset_btn = st.button("RESET", on_click=reset_fields)

    st.markdown("</div>", unsafe_allow_html=True)

# --- PREDICTION + RESULT (RIGHT) ---
with col2:
    st.markdown("<div style='margin-left: 40%; color: white;'>", unsafe_allow_html=True)
    st.markdown("<h3>PREDICTION</h3>", unsafe_allow_html=True)

    if predict_btn:
        st.session_state.reset_flag = False  # Clear reset mode

        # Check and convert inputs
        inputs = {}
        fields = {
            "air_temp": "Air temperature",
            "process_temp": "Process temperature",
            "rotation_speed": "Rotational speed",
            "torque": "Torque",
            "tool_wear": "Tool wear"
        }

        error_flag = False
        for key, label in fields.items():
            val = st.session_state.get(key, "").strip()
            if val == "":
                st.error(f"{label} is required.")
                error_flag = True
            else:
                try:
                    inputs[key] = float(val)
                except ValueError:
                    st.error(f"{label} must be a number.")
                    error_flag = True

        if not error_flag:
            input_list = list(inputs.values())
            input_scaled = scaler.transform([input_list])
            prediction = rf_model.predict(input_scaled)[0]

            st.session_state.previous_data.append({
                'input': input_list,
                'prediction': prediction
            })

    # Show results
    if st.session_state.previous_data:
        df_history = pd.DataFrame([
            {
                'Air temperature [K]': d['input'][0],
                'Process temperature [K]': d['input'][1],
                'Rotational speed [rpm]': d['input'][2],
                'Torque [Nm]': d['input'][3],
                'Tool wear [min]': d['input'][4],
                'Failure': 'No' if d['prediction'] == 0 else 'Yes'
            }
            for d in st.session_state.previous_data
        ])

        for data in st.session_state.previous_data[::-1]:
            air_temp_val, process_temp_val, rotation_speed_val, torque_val, tool_wear_val = data['input']
            prediction = data['prediction']
            box_class = "no-failure-box" if prediction == 0 else "failure-box"
            message = "NO FAILURE" if prediction == 0 else "FAILURE DETECTED"
            st.markdown(f"<div class='{box_class}'>{message}</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <table>
                <tr>
                    <th>Air temperature [K]</th>
                    <th>Process temperature [K]</th>
                    <th>Rotational speed [rpm]</th>
                    <th>Torque [Nm]</th>
                    <th>Tool wear [min]</th>
                    <th>Failure</th>
                </tr>
                <tr>
                    <td>{air_temp_val}</td>
                    <td>{process_temp_val}</td>
                    <td>{rotation_speed_val}</td>
                    <td>{torque_val}</td>
                    <td>{tool_wear_val}</td>
                    <td>{'✅' if prediction == 0 else '❌'}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        # Download & Clear buttons
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction History",
            data=csv,
            file_name='lrt_predictions.csv',
            mime='text/csv'
        )
        if st.button("🧹 Clear Prediction History"):
            st.session_state.previous_data = []
            st.success("Prediction history cleared.")

    st.markdown("</div>", unsafe_allow_html=True)
