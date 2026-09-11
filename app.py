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

# -------------------------------
# TITLE
# -------------------------------

st.title("🍱 Smart Canteen Food Demand Prediction")
st.write(
    "AI-based system to predict food demand and reduce food waste "
    "using machine learning."
)

st.divider()

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_excel("canteen_dataset.xlsx.xlsx")

# Encode categorical columns
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

# -------------------------------
# TRAIN MODEL
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
score = model.score(X_test, y_test)

# -------------------------------
# PROJECT INFORMATION
# -------------------------------

st.header("📌 About the System")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Records", len(df))

with col2:
    st.metric("ML Model", "Random Forest")

with col3:
    st.metric("R² Score", round(score, 2))

st.write(
    "The system learns from previous canteen data and predicts "
    "the expected food demand for a given situation."
)

st.divider()

# -------------------------------
# PREDICTION SECTION
# -------------------------------

st.header("🤖 Smart Food Demand Prediction")

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

    st.write("")
    predict_button = st.button(
        "🔮 Predict Food Demand",
        use_container_width=True
    )

# -------------------------------
# PREDICTION
# -------------------------------

if predict_button:

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

    # Decision logic
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
        st.metric("Model R² Score", round(score, 2))

    st.success(
        "💡 Smart Recommendation: " + recommendation
    )

# -------------------------------
# DEMAND ANALYSIS
# -------------------------------

st.divider()

st.header("📈 Demand Analysis")

chart_data = df[["Actual_Demand"]].copy()
chart_data["Record"] = range(1, len(chart_data) + 1)

fig, ax = plt.subplots()

ax.plot(
    chart_data["Record"],
    chart_data["Actual_Demand"],
    marker="o"
)

ax.set_xlabel("Dataset Record")
ax.set_ylabel("Food Demand")
ax.set_title("Food Demand Trend")

st.pyplot(fig)

# -------------------------------
# MODEL PERFORMANCE
# -------------------------------

st.header("🧠 Model Performance")

m1, m2 = st.columns(2)

with m1:
    st.metric(
        "Mean Absolute Error",
        round(mae, 2)
    )

with m2:
    st.metric(
        "R² Score",
        round(score, 2)
    )

st.write(
    "The Random Forest model uses student count, weather, "
    "previous sales, special events and exam days to estimate "
    "food demand."
)

# -------------------------------
# SMART WASTE MANAGEMENT
# -------------------------------

st.divider()

st.header("♻️ Smart Waste Management")

st.write(
    "The predicted demand can help canteen staff prepare an "
    "appropriate quantity of food instead of preparing excess food."
)

st.info(
    "Goal: Reduce food wastage while maintaining sufficient food availability."
)

# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "Smart Canteen AI | Machine Learning Based Food Demand Prediction"
)
