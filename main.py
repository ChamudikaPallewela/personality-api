from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load models
model_a = joblib.load("Random_Forest.joblib")
model_b = joblib.load("Logistic_Regression.joblib")

label_map = {1: "Introvert", 0: "Extrovert"}

class PersonalityInput(BaseModel):
    Time_spent_Alone: float
    Stage_fear: int
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: int
    Friends_circle_size: float
    Post_frequency : float

@app.post("/predict")
def predict(data: PersonalityInput):
    try:
        input_array = np.array([[
            data.Time_spent_Alone,
            data.Stage_fear,
            data.Social_event_attendance,
            data.Going_outside,
            data.Drained_after_socializing,
            data.Friends_circle_size,
            data.Post_frequency
        ]])

        # Predict with primary model
        prediction_a = model_a.predict(input_array)[0]
        prob_a = model_a.predict_proba(input_array)[0][prediction_a]

        if prob_a >= 0.6:
            model_used = "Random Forest"
            final_prediction = prediction_a
        else:
            # Fallback to Logistic Regression
            model_used = "Logistic Regression"
            final_prediction = model_b.predict(input_array)[0]

        return {
            "prediction": label_map[final_prediction],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
