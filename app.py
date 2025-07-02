import streamlit as st
import joblib # joblib installed

# Load the saved model, scaler, and label encoder
try:
    model = joblib.load('default_random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    st.success("Model and preprocessing objects loaded successfully.")
except FileNotFoundError:
    st.error("Error: Model or preprocessing files not found. Please ensure 'default_random_forest_model.joblib', 'scaler.joblib', and 'label_encoder.joblib' are in the same directory.")
except Exception as e:
    st.error(f"An error occurred while loading the files: {e}")

st.title("Maternal Health Risk Prediction")

# Add content to the Streamlit app here in subsequent steps
# For now, we just confirm loading
st.write("Application is ready to predict maternal health risk.")
