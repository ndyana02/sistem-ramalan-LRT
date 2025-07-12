
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and scaler
rf_model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sistem Ramalan Penyenggaraan Transit Aliran Ringan", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    body, .main {
        background-color: #6b2929;
    }
    .stApp {
    background-color: #6b2929;
}

    h1, h3, label {
        color: white !important;
    }
    .stTextInput > div > div > input, .stNumberInput > div > input {
        border-radius: 5px;
        width: 100%;
        box-sizing: border-box;
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
    }
    th {
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown("<h1 style='text-align: center;'>PREDICTIVE MAINTENANCE OF LIGHT RAIL TRANSIT (LRT)</h1>", unsafe_allow_html=True)

# --- LAYOUT ---
col1, col2 = st.columns([1, 2])

if 'previous_data' not in st.session_state:
    st.session_state.previous_data = []

# --- INPUT FORM (LEFT) ---
with col1:
    st.markdown("""
    <div class='input-box'>
        <h3>Enter Data</h3>
    """, unsafe_allow_html=True)

    air_temp = st.text_input("Air temperature [K]:", key="air_temp", value="")
    process_temp = st.text_input("Process temperature [K]:", key="process_temp", value="")
    rotation_speed = st.text_input("Rotational speed [rpm]:", key="rotation_speed", value="")
    torque = st.text_input("Torque [Nm]:", key="torque", value="")
    tool_wear = st.text_input("Tool wear [min]:", key="tool_wear", value="")

    col_submit, col_reset = st.columns(2)
    with col_submit:
        predict_btn = st.button("SUBMIT")
    with col_reset:
        reset_btn = st.button("RESET")

    if reset_btn:
        # Clear input fields only
        st.session_state.air_temp = ""
        st.session_state.process_temp = ""
        st.session_state.rotation_speed = ""
        st.session_state.torque = ""
        st.session_state.tool_wear = ""

    st.markdown("</div>", unsafe_allow_html=True)

# --- PREDICTION + RESULT (RIGHT) ---
with col2:
    # Wrap everything in a div with margin-left to push content right
    st.markdown("<div style='margin-left: 40%; color: white;'>", unsafe_allow_html=True)

    # Prediction Title
    st.markdown("<h3>PREDICTION</h3>", unsafe_allow_html=True)

    if predict_btn:
        try:
            input_data = [
                float(air_temp),
                float(process_temp),
                float(rotation_speed),
                float(torque),
                float(tool_wear)
            ]

            input_scaled = scaler.transform([input_data])
            prediction = rf_model.predict(input_scaled)[0]

            st.session_state.previous_data.append({
                'input': input_data,
                'prediction': prediction
            })
        except ValueError:
            st.error("Please enter valid numeric values in all fields.")

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

            result_box = f"""
            <div class='{box_class}'>{message}</div>
            """
            st.markdown(result_box, unsafe_allow_html=True)

            table_html = f"""
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
            """
            st.markdown(table_html, unsafe_allow_html=True)

        # Download button
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction History",
            data=csv,
            file_name='lrt_predictions.csv',
            mime='text/csv'
        )

    st.markdown("</div>", unsafe_allow_html=True)
