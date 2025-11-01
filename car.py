# ==============================
# 📌 Import Libraries
# ==============================
import streamlit as st
import pandas as pd
import pickle

# ML utilities (agar training code future me add karna ho)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==============================
# 📌 Load Trained Model & Encoders
# ==============================
with open("model_car.pkl", "rb") as f:
    model = pickle.load(f)   # Trained ML model
with open("car_price_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)   # Saved LabelEncoders for categorical data


# ==============================
# 📌 Streamlit App Title
# ==============================
st.title("🚗 Car Price Prediction App")


# ==============================
# 📌 User Input Section
# ==============================

# City where car is registered
registered = st.selectbox("Car registered in:", options=encoders["registered"].classes_)

# Car model year (numeric input)
car_model = st.number_input("Enter Car Model (e.g., 2014, 2018):", min_value=1990, max_value=2025, value=2015)

# Car mileage (numeric input)
mileage = st.number_input("Enter Car Mileage (km):", min_value=0, value=50000, step=500)

# Fuel type
fuel_type = st.selectbox("Select Fuel Type:", options=encoders["fuel_type"].classes_)

# Transmission type
transmission = st.selectbox("Select Transmission Type:", options=encoders["transmission"].classes_)

# Car color
color = st.selectbox("Select Car Color:", options=encoders["color"].classes_)

# Assembly type (Local / Imported)
assembly = st.selectbox("Select Assembly Type:", options=encoders["assembly"].classes_)

# Engine capacity in cc
engine_capacity = st.number_input("Enter Engine Capacity (cc):", min_value=600, max_value=6000, value=1600, step=100)

# Car age in years
vehicle_age = st.number_input("Enter Car Age (years):", min_value=0, max_value=50, value=1)


# ==============================
# 📌 Encode Categorical Features
# ==============================
registered_enc   = encoders["registered"].transform([registered])[0]
fuel_type_enc    = encoders["fuel_type"].transform([fuel_type])[0]
transmission_enc = encoders["transmission"].transform([transmission])[0]
color_enc        = encoders["color"].transform([color])[0]
assembly_enc     = encoders["assembly"].transform([assembly])[0]


# ==============================
# 📌 Prepare Data for Prediction
# ==============================
X_new = pd.DataFrame([[
    car_model, mileage, fuel_type_enc,
    transmission_enc, registered_enc, color_enc, assembly_enc,
    engine_capacity, vehicle_age
]], columns=[
    "model", "mileage", "fuel_type",
    "transmission", "registered", "color", "assembly",
    "engine_capacity", "vehicle_age"
])


# ==============================
# 📌 Helper Function to Format Price
# ==============================
def format_price(value):
    """Convert large numbers into Lakh, Crore, or Million format."""
    if value >= 10**7:      # Crore
        return f"{value/10**7:.2f} Crore"
    elif value >= 10**5:    # Lakh
        return f"{value/10**5:.2f} Lakh"
    elif value >= 10**6:    # Million
        return f"{value/10**6:.2f} Million"
    else:
        return f

# ==============================
# 📌 Prediction Button
# ==============================
if st.button("Predict Price"):
    prediction = model.predict(X_new)
    price = prediction[0]

    # Show price in numbers + words (e.g. PKR 5,200,000 (52 Lakh))
    st.success(f"Estimated Car Price: PKR {price:,.0f} ({format_price(price)})")