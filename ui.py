import streamlit as st
from api.constants import *
import requests



with st.form("my_form"):

    fixed_acidity = st.slider(label="fixed acidity", min_value=min_fixed_acidity, max_value=max_fixed_acidity)
    volatile_acidity = st.slider(label="volatile acidity", min_value=min_volatile_acidity, max_value=max_volatile_acidity)
    citric_acid = st.slider(label="citric acid", min_value=min_citric_acid, max_value=max_citric_acid)
    residual_sugar = st.slider(label="residual sugar", min_value=min_residual_sugar, max_value=max_residual_sugar)
    chlorides = st.slider(label="chlorides", min_value=min_chlorides, max_value=max_chlorides)
    free_sulfur_dioxide = st.slider(label="free sulfur dioxide", min_value=min_free_sulfur_dioxide, max_value=max_free_sulfur_dioxide)
    total_sulfur_dioxide = st.slider(label="total sulfur dioxide", min_value=min_total_sulfur_dioxide, max_value=max_total_sulfur_dioxide)
    density = st.slider(label="density", min_value=min_density, max_value=max_density)
    pH = st.slider(label="pH", min_value=min_pH, max_value=max_pH)
    sulphates = st.slider(label="sulphates", min_value=min_sulphates, max_value=max_sulphates)
    alcohol = st.slider(label="alcohol", min_value=min_alcohol, max_value=max_alcohol)

    submitted = st.form_submit_button('Submit')

if submitted:
    payload = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol       
    }

    try:
        API_URL = "http://localhost:8000/predict"
        with st.spinner("Fetching prediction from model..."):
            response = requests.post(url=API_URL, json=payload)
            response.raise_for_status()

            result = response.json()
            prediction = result.get("prediction", "")

        st.balloons()
        st.success(f"### Predicted Quality Score: {prediction}")
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
