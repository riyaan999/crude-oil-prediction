from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pickle

# Load the ARIMA model from the pickle file
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def main():
    # Create the Streamlit user interface
    st.title("Crude Oil Price Prediction using ARIMA")

    # Input data
    start = st.date_input("Start date", datetime(2024, 1, 1), min_value=datetime.now())
    end = start + timedelta(days=30)

    # Predict future values
    index_future_dates = pd.date_range(start=start, end=end)
    prediction = model.predict(start=1, end=len(index_future_dates), typ="levls")
    prediction.index = index_future_dates

    # Display the predictions
    st.write(f"Predicted Crude Oil prices between {start} and {end} are")
    st.line_chart(prediction)


if __name__ == "__main__":
    main()
