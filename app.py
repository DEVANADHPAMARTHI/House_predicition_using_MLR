from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------
df = pd.read_csv("data.csv")

# Create city mapping exactly like training
cities = df["city"].unique()

city_map = {}

for i, city in enumerate(cities):
    city_map[city] = i


# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------
with open("MLResults.pkl", "rb") as f:
    model = pickle.load(f)


# ---------------------------------------------------
# Feature order used during training
# ---------------------------------------------------
FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "city",
    "country",
    "year",
    "month",
    "day"
]


# ---------------------------------------------------
# Home page
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        cities=list(city_map.keys())
    )


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])
        sqft_living = float(request.form["sqft_living"])
        sqft_lot = float(request.form["sqft_lot"])
        floors = float(request.form["floors"])
        waterfront = float(request.form["waterfront"])
        view = float(request.form["view"])
        condition = float(request.form["condition"])
        sqft_above = float(request.form["sqft_above"])
        sqft_basement = float(request.form["sqft_basement"])
        yr_built = float(request.form["yr_built"])
        yr_renovated = float(request.form["yr_renovated"])

        city_name = request.form["city"]
        city = city_map[city_name]

        # Your original code sets country = 0
        country = 0

        year = int(request.form["year"])
        month = int(request.form["month"])
        day = int(request.form["day"])

        # ------------------------------------------------
        # Create input DataFrame
        # ------------------------------------------------
        input_data = pd.DataFrame([[
            bedrooms,
            bathrooms,
            sqft_living,
            sqft_lot,
            floors,
            waterfront,
            view,
            condition,
            sqft_above,
            sqft_basement,
            yr_built,
            yr_renovated,
            city,
            country,
            year,
            month,
            day
        ]], columns=FEATURES)

        # ------------------------------------------------
        # Prediction
        # ------------------------------------------------
        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction=f"${prediction:,.2f}",
            cities=list(city_map.keys())
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e),
            cities=list(city_map.keys())
        )


# ---------------------------------------------------
# Run application
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)