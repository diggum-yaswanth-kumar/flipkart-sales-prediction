import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load label encoders
with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

st.title("Flipkart Sales Prediction App")

# Input fields based on training features
product_name = st.selectbox("Product Name", label_encoders['Product Name'].classes_)
category = st.selectbox("Category", label_encoders['Category'].classes_)
quantity = st.number_input("Quantity", min_value=1, value=1)
price = st.number_input("Price (₹)", min_value=1, value=1000)
customer_segment = st.selectbox("Customer Segment", label_encoders['Customer Segment'].classes_)

# Predict button
if st.button("Predict Revenue"):
    # Encode categorical inputs
    encoded_product = label_encoders['Product Name'].transform([product_name])[0]
    encoded_category = label_encoders['Category'].transform([category])[0]
    encoded_segment = label_encoders['Customer Segment'].transform([customer_segment])[0]

    # Build input DataFrame
    new_data = pd.DataFrame({
        'Product Name': [encoded_product],
        'Category': [encoded_category],
        'Quantity': [quantity],
        'Price': [price],
        'Customer Segment': [encoded_segment]
    })

    # Predict
    prediction = model.predict(new_data)
    st.success(f"Predicted Revenue: ₹{prediction[0]:,.2f}")
