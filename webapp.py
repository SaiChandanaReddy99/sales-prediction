import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = "Dataset/Cleaned_Advertising_Data.csv"
df = pd.read_csv(file_path)

# Train the model
X = df[['TV_Budget', 'Radio_Budget', 'Newspaper_Budget']]
y = df['Sales']
model = LinearRegression()
model.fit(X, y)

# Streamlit App
st.title("Advertising Sales Prediction App")
st.write("Enter the advertising budget values to predict sales.")

# Input fields for budget values
tv_budget = st.number_input("TV Budget ($)", min_value=0.0, format="%.2f")
radio_budget = st.number_input("Radio Budget ($)", min_value=0.0, format="%.2f")
newspaper_budget = st.number_input("Newspaper Budget ($)", min_value=0.0, format="%.2f")

# Prediction button
if st.button("Predict Sales"):
    input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
    predicted_sales = model.predict(input_data)[0]
    st.success(f"Predicted Sales: ${predicted_sales:.2f}")
