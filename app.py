import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load model and scaler
model = load_model('stock_rnn_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app layout
st.title("📈 Stock Price Predictor")

# User input
price1 = st.number_input("Enter Close Price 1:")
price2 = st.number_input("Enter Close Price 2:")
price3 = st.number_input("Enter Close Price 3:")

if st.button("Predict"):
    new_close_prices = np.array([[price1], [price2], [price3]])
    scaled_input = scaler.transform(new_close_prices)

    sequence_length = 60
    last_scaled_price = scaled_input[-1]
    X_new = np.repeat(last_scaled_price, sequence_length).reshape(1, sequence_length, 1)

    predicted_scaled_price = model.predict(X_new)
    predicted_price = scaler.inverse_transform(predicted_scaled_price)

    st.success(f"📊 Predicted Stock Price: ₹ {predicted_price[0][0]:.2f}")
