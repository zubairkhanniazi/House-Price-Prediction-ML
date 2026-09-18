import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("house_price_model.pkl", "rb"))

# Title
st.title("🏠 House Price Prediction")

st.write("Enter house details to predict the price")

# Inputs
area = st.number_input("House Area", min_value=500, max_value=10000, value=1500)

bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)

bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)

age = st.number_input("House Age", min_value=0, max_value=100, value=5)

# Prediction
if st.button("Predict Price"):

    input_data = np.array([[area, bedrooms, bathrooms, age]])

    prediction = model.predict(input_data)

    st.success(f"Estimated House Price: {prediction[0]:,.2f}")