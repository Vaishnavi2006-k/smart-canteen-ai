import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Smart Canteen AI",
    page_icon="🍱",
    layout="wide"
)

# Load dataset
df = pd.read_excel("canteen_dataset.xlsx.xlsx")

# Encode categorical data
weather_encoder = LabelEncoder()
event_encoder = LabelEncoder()
exam_encoder = LabelEncoder()

df["Weather"] = weather_encoder.fit_transform(df["Weather"])
df["Special_Event"] = event_encoder.fit_transform(df["Special_Event"])
df["Exam_Day"] = exam_encoder.fit_transform(df["Exam_Day"])

# Features and target
X = df[[
    "Student",
    "Weather",
    "Previous_Sales",
    "Special_Event",
    "Exam_Day"
]]

y = df["Actual_Demand"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
score = model.score(X_test, y_test)

# Sidebar
st.sidebar.title("🍱 Smart Canteen AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🤖 Prediction",
        "📊 Demand Analysis",
        "🧠 Model Performance",
        "♻️ Waste Management"
    ]
)

# HOME
if page == "🏠 Home":

    st.title("🍱 Smart Canteen Food Demand Prediction")

    st.write(
        "AI-based system to predict food demand and reduce food waste "
        "using Machine Learning."
    )

    st.divider()

    st.header("📌 Project Overview")

    st.write(
        "The system uses previous canteen data to predict the expected "
        "food demand. This helps canteen staff prepare an appropriate "
        "quantity of food and reduce unnecessary wastage."
    )


    with col1:
        st.metric("Dataset Records", len(df))

    with col2:
        st.metric("ML Model", "Random Forest")

    with col3:
        st.metric("R² Score", round(score, 2))

    st.divider()

    st.header("🔄 How It Works")

    st.write(
        "Student Information + Weather + Previous Sales + Events "
        "→ Machine Learning Model → Predicted Food Demand "
        "→ Waste Reduction Recommendation"
    )

    st.info(
        "🎯 Main Goal: Predict demand accurately and reduce food waste."
    )


# PREDICTION
elif page == "🤖 Prediction":

    st.title("🤖 Smart Food Demand Prediction")

    st.write(
        "Enter today's canteen information to predict the required food quantity."
    )

    col1, col2 = st.columns(2)

    with col1:

        student = st.number_input(
            "Number of Students",
            min_value=1,
            value=150
        )

        weather = st.selectbox(
            "Weather",
            ["Cloudy", "Rainy", "Sunny"]
        )

        previous_sales = st.number_input(
            "Previous Sales",
            min_value=0,
            value=140
        )

    with col2:

        special_event = st.selectbox(
            "Special Event",
            ["No", "Yes"]
        )

        exam_day = st.selectbox(
            "Exam Day",
            ["No", "Yes"]
        )

    if st.button("🔮 Predict Food Demand", use_container_width=True):

        weather_map = {
            "Cloudy": 0,
            "Rainy": 1,
            "Sunny": 2
        }

        event_map = {
            "No": 0,
            "Yes": 1
        }

        new_data = pd.DataFrame({
            "Student": [student],
            "Weather": [weather_map[weather]],
            "Previous_Sales": [previous_sales],
            "Special_Event": [event_map[special_event]],
            "Exam_Day": [event_map[exam_day]]
        })

        demand = round(model.predict(new_data)[0])

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

        st.divider()

        st.header("📊 Prediction Results")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric("Predicted Demand", demand)

        with r2:
            st.metric("Expected Waste", waste)

        with r3:
            st.metric("Preparation Level", preparation)

        with r4:
            st.metric("R² Score", round(score, 2))

        st.success(
            "💡 Smart Recommendation: " + recommendation
        )


# DEMAND ANALYSIS
elif page == "📊 Demand Analysis":

    st.title("📊 Demand Analysis")

    st.write(
        "Analysis of food demand patterns in the available canteen dataset."
    )

    col1, col2
