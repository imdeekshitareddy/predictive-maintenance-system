from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ----------------------------
# Load simplified model
# ----------------------------
MODEL_PATH = "models_simplified_v1"

model = joblib.load(f"{MODEL_PATH}/random_forest_simple.pkl")
feature_names = joblib.load(f"{MODEL_PATH}/feature_names_simple.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    input_data = {
        "Temperature (°C)": float(request.form["temperature"]),
        "Vibration (m/s²)": float(request.form["vibration"]),
        "Power (W)": float(request.form["power"]),
        "Voltage (V)": float(request.form["voltage"]),
        "Humidity (%)": float(request.form["humidity"]),
        "time_since_maintenance": float(request.form["maintenance"])
    }

    df = pd.DataFrame([input_data])

    probability = model.predict_proba(df)[0][1]
    prediction = "⚠️ Maintenance Required" if probability > 0.3 else "✅ Machine Healthy"

    return render_template(
        "index.html",
        prediction_text=prediction,
        probability=round(probability, 3)
    )


if __name__ == "__main__":
    app.run(debug=True)
