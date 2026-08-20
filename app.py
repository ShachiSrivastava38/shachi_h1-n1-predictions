import os
import logging
import joblib
from flask import Flask, jsonify, render_template_string, request, url_for
import pandas as pd
import urllib.request

app = Flask(__name__)

# Configure basic logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load the trained RandomForest model
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")  # Rename your .pkl file to model.pkl or adjust path
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    model = None
    logger.exception("Error loading model from %s: %s", MODEL_PATH, e)

# Ensure static directory exists and a vaccine image is present for the UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
VACCINE_IMAGE_PATH = os.path.join(STATIC_DIR, "vaccine.png")
if not os.path.exists(VACCINE_IMAGE_PATH):
    try:
        # Public domain / Wikimedia image as a fallback. Replace with your licensed asset if you have one.
        urllib.request.urlretrieve(
            "https://upload.wikimedia.org/wikipedia/commons/6/6a/Vaccine_injection.jpg",
            VACCINE_IMAGE_PATH,
        )
        logger.info("Downloaded default vaccine image to %s", VACCINE_IMAGE_PATH)
    except Exception:
        logger.exception("Failed to download default vaccine image; UI will show broken image if none provided")

# Explicit feature list expected by the model
FEATURE_NAMES = [
    "Unique_Id",
    "H1N1_Worry",
    "H1N1_Awareness",
    "Antiviral_Medication",
    "Contact_Avoidance",
    "Bought_Face_Mask",
    "Wash_Hands_Frequently",
    "Avoid_Large_Gatherings",
    "Reduced_Outside_Home_Cont",
    "Avoid_Touch_Face",
    "Dr_Recc_H1N1_Vacc",
    "Dr_Recc_Seasonal_Vacc",
    "Chronic_Medic_Condition",
    "Cont_Child_Undr_6_Mnths",
    "Is_Health_Worker",
    "Has_Health_Insur",
    "Is_H1N1_Vacc_Effective",
    "Is_H1N1_Risky",
    "Sick_From_H1N1_Vacc",
    "Is_Seas_Vacc_Effective",
    "Is_Seas_Risky",
    "Sick_From_Seas_Vacc",
    "Age_Bracket",
    "Qualification",
    "Race",
    "Sex",
    "Income_Level",
    "Marital_Status",
    "Housing_Status",
    "Employment",
    "Census_Msa",
    "No_Of_Adults",
    "No_Of_Children",
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>H1N1 Vaccine Prediction API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; }
        .container { max-width: 900px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h2 { color: #333; }
        .form-group { margin-bottom: 12px; }
        label { display: inline-block; width: 220px; font-weight: bold; font-size: 0.9em; }
        input { padding: 6px; width: 200px; }
        button { margin-top: 15px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; font-weight: bold; font-size: 1.1em; color: #28a745; }
        .header { display:flex; align-items:center; justify-content:space-between; }
        .header img { max-width:160px; border-radius:8px; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h2>H1N1 Vaccine Model Inference</h2>
        <!-- Vaccine image; place a file named static/vaccine.png in the repository (or the app will download a default) -->
        <img src="{{ url_for('static', filename='vaccine.png') }}" alt="Vaccine Image" />
    </div>
    <form method="POST" action="/predict">
        {% for feature in features %}
        <div class="form-group">
            <label for="{{ feature }}">{{ feature }}:</label>
            <input type="text" id="{{ feature }}" name="{{ feature }}" value="0" required>
        </div>
        {% endfor %}
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
    <div class="result">
        Prediction: {{ prediction }}
    </div>
    {% endif %}
</div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    try:
        # Check if request is JSON or Web Form
        if request.is_json:
            data = request.get_json()
            df = pd.DataFrame([data])
        else:
            data = request.form.to_dict()
            df = pd.DataFrame([data])

        # Ensure all columns exist and maintain exact order
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0

        df = df[FEATURE_NAMES]

        # Convert object columns to numeric where possible
        df = df.apply(pd.to_numeric, errors="ignore")

        # Get prediction
        prediction = model.predict(df)[0]

        if request.is_json:
            return jsonify({"prediction": int(prediction)})

        return render_template_string(
            HTML_TEMPLATE, features=FEATURE_NAMES, prediction=int(prediction)
        )

    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Simple health endpoint for load balancers and monitoring."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if model is not None else None,
    })


if __name__ == "__main__":
    # The werkzeug reloader registers signal handlers which may fail in some hosting
    # environments or when running the app from a non-main thread. Disable the reloader
    # to avoid ValueError related to signal handling.
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug, use_reloader=False)
