import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# Load the trained pipeline
with open("weather_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Page setup
st.set_page_config(page_title="Weather Predictor Dashboard", page_icon="🌦️", layout="wide")

# Sidebar for inputs
st.sidebar.header("🌡️ Enter Weather Conditions")
temp = st.sidebar.number_input("Temperature (°C)", -30.0, 50.0, 20.0)
humidity = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)
wind_bearing = st.sidebar.number_input("Wind Bearing (degrees)", 0.0, 360.0, 180.0)
visibility = st.sidebar.number_input("Visibility (km)", 0.0, 20.0, 10.0)
pressure = st.sidebar.number_input("Pressure (millibars)", 900.0, 1100.0, 1013.0)

# Main dashboard title
st.title("🌦️ Weather Precipitation Type Predictor")
st.write("This app predicts whether the weather will be **Rain, Snow, or Clear** using machine learning.")

# Predict button
if st.sidebar.button("🔮 Predict"):
    features = np.array([[temp, humidity, wind_speed, wind_bearing, visibility, pressure]])
    prediction = pipeline.predict(features)[0]
    probabilities = pipeline.predict_proba(features)[0]

    # Layout with two columns
    col1, col2 = st.columns([1, 2])

    # Column 1: Prediction result
    with col1:
        st.subheader("✅ Prediction Result")
        st.success(f"Predicted Precipitation: **{prediction}**")

        # Probability table
        prob_df = pd.DataFrame({
            "Precipitation Type": pipeline.classes_,
            "Probability (%)": (probabilities * 100).round(2)
        })
        st.subheader("📊 Prediction Probabilities")
        st.table(prob_df)

    # Column 2: Interactive Plotly Bar Chart
    with col2:
        st.subheader("📈 Probability Distribution")
        fig = px.bar(
            prob_df,
            x="Precipitation Type",
            y="Probability (%)",
            color="Precipitation Type",
            text="Probability (%)",
            title="Prediction Probability Distribution",
            color_discrete_sequence=["skyblue", "orange", "lightgreen"]
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(yaxis=dict(title="Probability (%)"), xaxis=dict(title="Precipitation Type"))
        st.plotly_chart(fig, use_container_width=True)
