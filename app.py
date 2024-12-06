import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the trained model

# Load your pre-trained model
# C:\Users\acer\Downloads\financial_planning_data2.pkl
MODEL_PATH = r"C:\Users\acer\Downloads\financial_planning_data2.pkl"  # Update with the correct path
model = joblib.load(MODEL_PATH)

# Title of the app
st.title("AI for Personalized Financial Planning")

# Sidebar for user inputs
st.sidebar.header("User Inputs")

# Collect user inputs
income = st.sidebar.number_input("Enter your monthly income ($)", min_value=1000, max_value=500000, step=1000)
expenses = st.sidebar.number_input("Enter your monthly expenses ($)", min_value=500, max_value=300000, step=500)
investment = st.sidebar.number_input("Enter your monthly investment ($)", min_value=0, max_value=100000, step=100)

# Submit button
if st.sidebar.button("Submit"):
    # Prepare user input for the model
    user_data = np.array([[income, expenses, investment]])

    # Predict using the pre-trained model
    prediction = model.predict(user_data)[0]

    # Display the result
    st.subheader("Prediction Result")
    st.write(f"Based on your inputs, the model predicts: **{prediction}**")

    # Add tailored advice based on prediction
    if prediction == "Low":
        st.write("✅ Consider low-risk investments such as bonds or fixed deposits.")
    elif prediction == "Medium":
        st.write("✅ Explore balanced mutual funds or a mix of stocks and bonds.")
    elif prediction == "High":
        st.write("✅ High-risk investments such as equity stocks or cryptocurrencies may suit your profile.")

    # Display input data as a table
    st.subheader("Your Financial Data")
    user_df = pd.DataFrame({
        "Income": [income],
        "Expenses": [expenses],
        "Investment": [investment],
        "Predicted Risk Tolerance": [prediction]
    })
    st.table(user_df)
