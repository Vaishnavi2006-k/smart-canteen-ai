import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Smart Canteen AI", page_icon="🍱", layout="wide")

# Load dataset
df = pd.read_excel("canteen_dataset.xlsx.xlsx")

# Encode columns
df["Weather"] = LabelEncoder().fit_transform(df["Weather"])
df["Special_Event"] = LabelEncoder().fit_transform(df["Special_Event"])
df["Exam_Day"] = LabelEncoder().fit_transform(df["Exam_Day"])

# Features and target
X = df[["Student", "Weather", "Previous_Sales", "Special_Event", "Exam_Day"]]
y = df["Actual_Demand"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
score = model.score(X_test, y_test)

# Store latest prediction
if "latest_demand" not in st.session_state:
     st.session_state.latest_demand = None

# Prediction History
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
    
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
        "The system learns from previous canteen data and predicts "
        "the expected food demand. This helps canteen staff prepare "
        "the appropriate quantity of food and reduce unnecessary wastage."
    )

    st.subheader("🔄 How It Works")

    st.write(
        "Student Count + Weather + Previous Sales + Special Event + Exam Day "
        "→ Random Forest Model → Predicted Demand → Waste Recommendation"
    )

    st.divider()

    st.header("📊 Project Information")

    st.write("**Dataset Records:**", len(df))
    st.write("**Machine Learning Model:** Random Forest Regressor")
    st.write("**R² Score:**", round(score, 2))
    st.write("**Mean Absolute Error:**", round(mae, 2))

    st.info(
        "🎯 Goal: Predict food demand and help reduce food wastage."
    )


# PREDICTION
elif page == "🤖 Prediction":

    st.title("🤖 Smart Food Demand Prediction")

    st.write("Enter canteen information to predict food demand.")

    student = st.number_input(
        "Number of Students", min_value=1, value=150
    )

    weather = st.selectbox(
        "Weather", ["Cloudy", "Rainy", "Sunny"]
    )

    previous_sales = st.number_input(
        "Previous Sales", min_value=0, value=140
    )

    special_event = st.selectbox(
        "Special Event", ["No", "Yes"]
    )

    exam_day = st.selectbox(
        "Exam Day", ["No", "Yes"]
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
        st.session_state.latest_demand = demand
        st.session_state.prediction_history.append({
             "Students": student,
             "Weather": weather,
             "Previous Sales": previous_sales,
             "Special Event": special_event,
             "Exam Day": exam_day,
             "Predicted Demand": demand
        })

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

        st.metric("Predicted Food Demand", demand)
        st.metric("Expected Waste Level", waste)
        st.metric("Food Preparation Level", preparation)

        st.success(
            "💡 " + recommendation
        )


# DEMAND ANALYSIS
elif page == "📊 Demand Analysis":

    st.title("📊 Demand Analysis")

    st.write(
        "Food demand pattern from the available canteen dataset."
    )

    st.metric(
        "Average Food Demand",
        round(df["Actual_Demand"].mean(), 1)
    )

    st.metric(
        "Maximum Food Demand",
        int(df["Actual_Demand"].max())
    )

    st.divider()

    chart_data = df["Actual_Demand"].reset_index()
    chart_data.columns = ["Record", "Demand"]

    fig, ax = plt.subplots()

    ax.plot(
        chart_data["Record"],
        chart_data["Demand"],
        marker="o"
    )

    ax.set_xlabel("Dataset Record")
    ax.set_ylabel("Food Demand")
    ax.set_title("Food Demand Trend")

    st.pyplot(fig)

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# MODEL PERFORMANCE
elif page == "🧠 Model Performance":

    st.title("🧠 Model Performance")

    st.metric("R² Score", round(score, 2))

    st.metric("Mean Absolute Error", round(mae, 2))

    st.metric("Training Records", len(X_train))

    st.divider()

    st.header("🌳 Random Forest Regressor")

    st.write(
        "Random Forest combines multiple decision trees to make "
        "more reliable predictions."
    )

    st.write(
        "The model uses Student count, Weather, Previous Sales, "
        "Special Event and Exam Day as input features."
    )

    st.info(
        "The model learns patterns from historical canteen data "
        "to estimate future food demand."
    )


# WASTE MANAGEMENT
elif page == "♻️ Waste Management":

    st.title("♻️ Smart Waste Reduction System")

    st.write(
        "The system recommends an appropriate food preparation quantity "
        "based on predicted demand to reduce unnecessary food waste."
    )

    st.divider()

    st.subheader("📊 Latest Prediction")

    predicted_demand = st.session_state.latest_demand

    if predicted_demand is None:

        st.warning(
            "Please make a prediction first from the Prediction page."
        )

    else:

        st.info(
            f"Latest Predicted Demand: {predicted_demand}"
        )

        if st.button(
            "♻️ Generate Waste Reduction Plan",
            use_container_width=True
        ):

            recommended_quantity = predicted_demand + 5

            if predicted_demand < 100:
                waste_level = "Low"
                suggestion = (
                    "Prepare a small quantity and avoid overproduction."
                )

            elif predicted_demand < 160:
                waste_level = "Medium"
                suggestion = (
                    "Prepare a moderate quantity close to predicted demand."
                )

            else:
                waste_level = "High"
                suggestion = (
                    "Prepare food in batches and monitor demand carefully."
                )

            st.divider()

            st.subheader("🎯 Waste Reduction Results")

            st.metric(
                "Predicted Demand",
                predicted_demand
            )

            st.metric(
                "Recommended Preparation",
                recommended_quantity
            )

            st.metric(
                "Expected Waste Level",
                waste_level
            )

            st.success(
                "💡 " + suggestion
            )
            
st.sidebar.divider()

st.sidebar.caption(
    "Smart Canteen AI | Machine Learning Project"
)
