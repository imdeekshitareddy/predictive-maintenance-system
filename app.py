from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ------------------------
# Load model artifacts
# ------------------------
MODEL_PATH = "models"

model = joblib.load(f"{MODEL_PATH}/random_forest_model.pkl")
feature_names = joblib.load(f"{MODEL_PATH}/feature_names.pkl")

# scaler optional
try:
    scaler = joblib.load(f"{MODEL_PATH}/standard_scaler.pkl")
except:
    scaler = None


@app.route("/")
def home():
    return "Predictive Maintenance API Running"


@app.route("/predict", methods=["POST"])
def predict():

    input_data = request.json

    # convert input to dataframe
    df = pd.DataFrame([input_data])

    # ensure correct feature order
    df = df[feature_names]

    # scale if needed
    if scaler is not None:
        df = scaler.transform(df)

    # prediction
    probability = model.predict_proba(df)[0][1]
    prediction = int(probability > 0.5)

    return jsonify({
        "fault_prediction": prediction,
        "failure_probability": float(probability)
    })


if __name__ == "__main__":
    app.run(debug=True)
