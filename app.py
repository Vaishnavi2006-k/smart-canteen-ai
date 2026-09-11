import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.title("Smart Canteen Food Demand Prediction")
st.write("AI-based system to predict canteen food demand and reduce food waste.")

df = pd.read_excel("canteen_dataset.xlsx.xlsx")

df["Weather"] = LabelEncoder().fit_transform(df["Weather"])
df["Special_Event"] = LabelEncoder().fit_transform(df["Special_Event"])
df["Exam_Day"] = LabelEncoder().fit_transform(df["Exam_Day"])

X = df[["Student", "Weather", "Previous_Sales", "Special_Event", "Exam_Day"]]
y = df["Actual_Demand"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

student = st.number_input("Number of Students", min_value=1, value=150)
weather = st.selectbox("Weather", ["Cloudy", "Rainy", "Sunny"])
previous_sales = st.number_input("Previous Sales", min_value=0, value=140)
special_event = st.selectbox("Special Event", ["No", "Yes"])
exam_day = st.selectbox("Exam Day", ["No", "Yes"])

if st.button("Predict Food Demand"):

    weather_map = {"Cloudy": 0, "Rainy": 1, "Sunny": 2}
    event_map = {"No": 0, "Yes": 1}

    data = pd.DataFrame({
        "Student": [student],
        "Weather": [weather_map[weather]],
        "Previous_Sales": [previous_sales],
        "Special_Event": [event_map[special_event]],
        "Exam_Day": [event_map[exam_day]]
    })

    demand = round(model.predict(data)[0])

    if demand < 100:
        recommendation = "Prepare Low quantity to reduce food waste."
        waste = "Low"
        preparation = "Low Preparation"
    elif demand < 160:
        recommendation = "Prepare Medium quantity according to predicted demand."
        waste = "Medium"
        preparation = "Medium Preparation"
    else:
        recommendation = "Prepare High quantity to meet expected demand."
        waste = "High"
        preparation = "High Preparation"

    st.success(f"Predicted Food Demand: {demand}")
    st.info(f"Waste Reduction Recommendation: {recommendation}")
    st.warning(f"Expected Waste Level: {waste}")
    st.write(f"Food Preparation Level: {preparation}")
