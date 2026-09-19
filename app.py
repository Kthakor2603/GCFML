import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Model pipeline container
model_pipeline = None

def train_or_load_model():
    global model_pipeline
    csv_path = os.path.join("data", "heart.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please create the 'data/heart.csv' file.")
    
    df = pd.read_csv(csv_path)

    # Features and Target
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"].astype(int)

    categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    numeric_features = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    # Full Random Forest pipeline matching the R script parameters
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=500, random_state=42))
        ]
    )

    model_pipeline.fit(X, y)
    print("Random Forest model trained and ready.")

# Train on startup
train_or_load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Build single-row DataFrame matching the exact dataset columns
        input_data = pd.DataFrame([{
            "Age": float(data["Age"]),
            "Sex": str(data["Sex"]),
            "ChestPainType": str(data["ChestPainType"]),
            "RestingBP": float(data["RestingBP"]),
            "Cholesterol": float(data["Cholesterol"]),
            "FastingBS": int(data["FastingBS"]),
            "RestingECG": str(data["RestingECG"]),
            "MaxHR": float(data["MaxHR"]),
            "ExerciseAngina": str(data["ExerciseAngina"]),
            "Oldpeak": float(data["Oldpeak"]),
            "ST_Slope": str(data["ST_Slope"])
        }])

        probabilities = model_pipeline.predict_proba(input_data)[0]
        # Class 1 is Heart Disease
        risk_percentage = round(probabilities[1] * 100, 2)
        prediction = 1 if risk_percentage >= 50 else 0

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "risk_score": risk_percentage,
            "message": "High Risk: Signs of Heart Failure detected." if prediction == 1 else "Normal: Low Risk detected."
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)