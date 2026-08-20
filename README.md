# 🏠 House Price Prediction using Machine Learning

A complete **Machine Learning web application** that predicts house prices based on property characteristics such as bedrooms, bathrooms, living area, location, construction year, waterfront availability, and more.

The project uses a **Linear Regression Machine Learning model** built with Scikit-learn and provides an interactive web interface using **Flask**.

---

## 📌 Project Overview

House prices depend on several factors, including:

* Number of bedrooms
* Number of bathrooms
* Living area
* Lot area
* Number of floors
* Waterfront availability
* Property condition
* View rating
* Basement area
* Year built
* Year renovated
* City
* Date of sale

This project analyzes these features and uses a **Linear Regression model** to predict the estimated market price of a house.

The trained model is integrated into a Flask web application where users can enter property details and receive a predicted house price.

---

# ✨ Features

* 🤖 Machine Learning-based house price prediction
* 📊 Linear Regression model
* 🏠 Interactive property configuration form
* 📍 City-based property prediction
* 📅 Date feature extraction
* 🌊 Waterfront selection
* 📈 Model training and evaluation
* 💰 Real-time predicted market price
* 🎨 Modern responsive dashboard UI
* ⚡ Flask backend integration
* 💾 Trained model saved using Pickle

---

# 🖥️ Application Preview

The application contains two main sections:

## 1️⃣ Property Configuration

Users can enter property information such as:

* Bedrooms
* Bathrooms
* Living area
* Lot size
* Number of floors
* Waterfront availability
* View rating
* Property condition
* Above-ground area
* Basement area
* Built year
* Renovation year
* City
* Sale year
* Sale month
* Sale day

## 2️⃣ Valuation Summary

After submitting the form, the application displays:

* Estimated Market Price
* Prediction status
* Confidence indicator
* Price per square foot section
* Area variance section
* Market insights dashboard

---

# 📊 Dataset Information

The dataset used in this project contains **4,600 records** and **16 columns**.

## Dataset Columns

| Column          | Description                                          |
| --------------- | ---------------------------------------------------- |
| `date`          | Date when the property was sold                      |
| `price`         | House price — Target Variable                        |
| `bedrooms`      | Number of bedrooms                                   |
| `bathrooms`     | Number of bathrooms                                  |
| `sqft_living`   | Living area in square feet                           |
| `sqft_lot`      | Lot size in square feet                              |
| `floors`        | Number of floors                                     |
| `waterfront`    | Indicates whether the property has waterfront access |
| `view`          | Property view rating                                 |
| `condition`     | Overall condition of the property                    |
| `sqft_above`    | Above-ground living area                             |
| `sqft_basement` | Basement area                                        |
| `yr_built`      | Year the house was built                             |
| `yr_renovated`  | Year the house was renovated                         |
| `city`          | City where the property is located                   |
| `country`       | Country of the property                              |

---

# 🔄 Data Preprocessing

Before training the Machine Learning model, the dataset is preprocessed.

## Date Conversion

The `date` column is converted into datetime format.

```python
df["date"] = pd.to_datetime(df["date"])
```

The date is then separated into:

* Year
* Month
* Day

```python
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
```

The original `date` column is removed after extracting these features.

```python
df = df.drop(["date"], axis=1)
```

---

## City Encoding

Since Machine Learning models require numerical input, the city names are converted into numerical values.

```python
cities = df["city"].unique()

city_map = {}

for i, city in enumerate(cities):
    city_map[city] = i

df["city"] = df["city"].map(city_map)
```

The same city mapping must be used when users select a city in the Flask application.

---

## Country Processing

In the current project implementation, the `country` column is converted to a numerical value.

```python
df["country"] = 0
```

This allows the Linear Regression model to work with numerical features.

---

# 🎯 Target Variable

The target variable is:

```text
price
```

The remaining processed columns are used as input features.

```python
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
```

Where:

* `X` → Input features
* `y` → House prices

---

# 🤖 Machine Learning Model

This project uses the **Linear Regression** algorithm from Scikit-learn.

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
```

Linear Regression attempts to find the relationship between the input features and house prices.

The model learns how different property characteristics affect the final property price.

---

# 📂 Train-Test Split

The dataset is divided into training and testing data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The project uses:

* **80% Training Data**
* **20% Testing Data**

---

# 🧠 Model Training

The model is trained using the following code:

```python
reg.fit(X_train, y_train)
```

After training, the model learns the relationship between property features and house prices.

---

# 📈 Model Evaluation

The project evaluates the model using prediction results from both training and testing data.

The model generates predictions using:

```python
train_predict_values = reg.predict(X_train)

test_predict_values = reg.predict(X_test)
```

The project calculates:

* Training performance
* Testing performance
* Root Mean Squared Error (RMSE)
* Prediction accuracy using an R²-style calculation

A lower RMSE indicates that the predicted house prices are closer to the actual house prices.

---

# 💾 Saving the Trained Model

After training, the Machine Learning model is saved using Python's Pickle module.

```python
import pickle

with open("MLResults.pkl", "wb") as f:
    pickle.dump(obj.reg, f)
```

The saved model can later be loaded into the Flask application.

```python
with open("MLResults.pkl", "rb") as f:
    model = pickle.load(f)
```

---

# 🌐 Web Application

The trained model is integrated with a Flask web application.

The user enters property details through the web interface.

The Flask backend:

1. Receives user input
2. Converts the input into numerical values
3. Maps the selected city
4. Creates a Pandas DataFrame
5. Sends the data to the trained model
6. Generates a predicted house price
7. Displays the result on the webpage

---

# 🗂️ Project Structure

```text
House-Price-Prediction/
│
├── app.py
│
├── House prediction.py
│
├── data.csv
│
├── MLResults.pkl
│
├── requirements.txt
│
├── README.md
│
└── templates/
    │
    └── index.html
```

### File Description

| File                   | Description                            |
| ---------------------- | -------------------------------------- |
| `app.py`               | Flask application backend              |
| `House prediction.py`  | Machine Learning model training code   |
| `data.csv`             | Dataset used for training              |
| `MLResults.pkl`        | Saved trained Linear Regression model  |
| `templates/index.html` | User interface for the web application |
| `requirements.txt`     | Required Python libraries              |
| `README.md`            | Project documentation                  |

---

# ⚙️ Technologies Used

The following technologies and libraries were used in this project:

### Programming Language

* Python

### Machine Learning

* Scikit-learn

### Data Analysis

* Pandas
* NumPy

### Web Framework

* Flask

### Model Serialization

* Pickle

### Frontend

* HTML
* CSS
* Jinja Templates

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd House-Price-Prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

---

## 3. Install Required Libraries

```bash
pip install pandas numpy scikit-learn flask
```

Or, if you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

# 📄 requirements.txt

Create a file named `requirements.txt` and add:

```text
Flask
pandas
numpy
scikit-learn
```

You can also generate the requirements file using:

```bash
pip freeze > requirements.txt
```

---

# ▶️ Running the Machine Learning Model

First, train the model:

```bash
python "House prediction.py"
```

This will:

1. Load the dataset
2. Preprocess the data
3. Split the dataset
4. Train the Linear Regression model
5. Evaluate the model
6. Save the trained model as:

```text
MLResults.pkl
```

---

# 🚀 Running the Flask Application

Run:

```bash
python app.py
```

The Flask development server will start.

Open your browser and visit:

```text
http://127.0.0.1:5000
```

You can now enter property details and click:

```text
RUN PREDICTIVE ANALYSIS
```

The application will display the predicted house price.

---

# 🔮 Prediction Workflow

```text
                 HOUSE DATASET
                       │
                       ▼
              DATA PREPROCESSING
                       │
                       ▼
              FEATURE ENGINEERING
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        DATE FEATURES        CITY ENCODING
             │                   │
             └─────────┬─────────┘
                       ▼
                TRAIN / TEST SPLIT
                       │
                       ▼
               LINEAR REGRESSION
                       │
                       ▼
                 TRAINED MODEL
                       │
                       ▼
                PICKLE MODEL FILE
                       │
                       ▼
                  FLASK BACKEND
                       │
                       ▼
                   WEB INTERFACE
                       │
                       ▼
              HOUSE PRICE PREDICTION
```

---

# 🏠 Example Prediction

A user can enter values such as:

```text
Bedrooms:        3
Bathrooms:       2.5
Living Area:     1850 sqft
Lot Area:        5000 sqft
Floors:          2
Waterfront:      No
View Rating:     0
Condition:       3
Above Area:      1850 sqft
Basement:        0 sqft
Built Year:      1998
Renovated:       0
City:            Seattle
Sale Year:       2014
Sale Month:      5
Sale Day:        2
```

After clicking **RUN PREDICTIVE ANALYSIS**, the trained model estimates the market price based on the input features.

---

# 🎨 User Interface

The web application provides a modern dashboard with:

* Property configuration panel
* Input validation
* City selection dropdown
* Machine Learning prediction
* Valuation summary
* Estimated market price display
* Responsive design for different screen sizes

The interface is built using HTML and CSS and is connected to the Flask backend using Jinja templates.

---

# 🧩 Key Concepts Demonstrated

This project demonstrates several important Data Science and Machine Learning concepts:

* Data loading
* Data preprocessing
* Feature engineering
* Datetime conversion
* Categorical data encoding
* Train-test splitting
* Linear Regression
* Model training
* Model evaluation
* RMSE calculation
* Model serialization with Pickle
* Flask deployment
* Frontend and backend integration

---

# 🔧 Future Improvements

The following improvements can be added in future versions:

* [ ] Use One-Hot Encoding for city values
* [ ] Preserve preprocessing objects using a Scikit-learn Pipeline
* [ ] Add additional Machine Learning algorithms
* [ ] Compare Linear Regression with Random Forest
* [ ] Compare Linear Regression with XGBoost
* [ ] Add model evaluation charts
* [ ] Add actual R² and RMSE values to the dashboard
* [ ] Dynamically calculate price per square foot
* [ ] Add interactive market insights
* [ ] Add data visualizations
* [ ] Deploy the application online
* [ ] Add Docker support
* [ ] Improve model accuracy using feature engineering

---

# ⚠️ Important Note

The city mapping used during training must remain the same when making predictions.

For example:

```python
city_map = {
    "Seattle": 0,
    "Kent": 1,
    "Shoreline": 2
}
```

The exact mapping depends on the order of cities in the dataset.

For a production-level application, it is recommended to save the preprocessing steps along with the model using a Scikit-learn Pipeline or by saving the `city_map` separately.

---

# 👨‍💻 Author

**Devanadh Pamarthi**

Data Science | Machine Learning | Python

---

# ⭐ Support

If you found this project useful:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements

---
# 🔗 Live Demo & Repository

## 🚀 Live Application

The House Price Prediction application is deployed on Render.

🔗 **Live Demo:https://house-predicition-using-mlr-3.onrender.com
```

> ⚠️ **Note:** The application is hosted using Render's free tier. The first request may take a few seconds because the server can take time to wake up.

---

## 💻 GitHub Repository

The complete source code, Machine Learning model, dataset, Flask application, and frontend files are available on GitHub.

🔗 **GitHub Repository:https://github.com/DEVANADHPAMARTHI/House_predicition_using_MLR**

---

## 🌐 Quick Links

| Resource                  | Link                                                      |
| ------------------------- | --------------------------------------------------------- |
| 🚀 Live Application       | [Open House Price Predictor](YOUR_RENDER_DEPLOYMENT_LINK) |
| 💻 GitHub Repository      | [View Source Code](YOUR_GITHUB_REPOSITORY_LINK)           |
| 🤖 Machine Learning Model | Linear Regression                                         |
| 🌐 Deployment Platform    | Render                                                    |
| 🐍 Backend Framework      | Flask                                                     |

---


# 📜 License

This project is created for educational and learning purposes.

You can add an appropriate open-source license, such as the MIT License, if you plan to allow others to reuse or modify the project.

---

## 🎯 Conclusion

This project demonstrates an end-to-end Machine Learning workflow, from **data preprocessing and feature engineering to model training, evaluation, model saving, and deployment through a Flask web application**.

It provides a practical example of how a Machine Learning model can be integrated into a user-friendly web application to solve a real-world problem: **estimating house prices based on property characteristics**.

⭐ **If you like this project, don't forget to star the repository!**
