
import pandas as pd
import gradio as gr
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_excel("canteen_dataset.xlsx.xlsx")

df["Weather"] = LabelEncoder().fit_transform(df["Weather"])
df["Special_Event"] = LabelEncoder().fit_transform(df["Special_Event"])
df["Exam_Day"] = LabelEncoder().fit_transform(df["Exam_Day"])

X = df[["Student", "Weather", "Previous_Sales", "Special_Event", "Exam_Day"]]
y = df["Actual_Demand"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

def final_prediction(student, weather, previous_sales, special_event, exam_day):

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

    return demand, recommendation, "Expected Waste: " + waste, preparation


demo = gr.Interface(
    fn=final_prediction,
    inputs=[
        gr.Number(label="Number of Students", value=150),
        gr.Dropdown(["Cloudy", "Rainy", "Sunny"], label="Weather", value="Sunny"),
        gr.Number(label="Previous Sales", value=140),
        gr.Dropdown(["No", "Yes"], label="Special Event", value="No"),
        gr.Dropdown(["No", "Yes"], label="Exam Day", value="No")
    ],
    outputs=[
        gr.Number(label="Predicted Food Demand"),
        gr.Textbox(label="Waste Reduction Recommendation"),
        gr.Textbox(label="Expected Waste Level"),
        gr.Textbox(label="Food Preparation Level")
    ],
    title="Smart Canteen Food Demand Prediction",
    description="AI-based system to predict canteen food demand and reduce food waste."
)

demo.launch(share=True)
