import streamlit as st
import requests

# API endpoint (FastAPI is running locally on port 8000)
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="AI vs Human Classifier", page_icon="🤖", layout="centered")

st.title("🤖 AI vs Human Text Classifier")
st.write("Enter text below to check if it's written by a Human or AI.")

# Text input box
user_input = st.text_area("✍️ Paste your text here:", height=200)

if st.button("Classify"):
    if user_input.strip():
        payload = {"text": user_input}
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            st.subheader(f"✅ Prediction: **{prediction}**")
            st.write("### Probabilities")
            st.json(probabilities)

            # Visualization
            st.bar_chart(probabilities)
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    else:
        st.warning("⚠️ Please enter some text before classifying.")
