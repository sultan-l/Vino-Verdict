import streamlit as st
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

@st.cache_data
def load_model():
    # Create directory if it doesn't exist
    if not os.path.exists('local_model'):
        os.makedirs('local_model')

    # URLs to your model and config
    url='https://vino-api-v2-766cav374q-an.a.run.app/predict'
    response = requests.get(url)
    with open("local_model/pytorch_model.bin", "wb") as f:
        f.write(response.content)
    
    url='https://storage.googleapis.com/vv-2/config.json'
    response = requests.get(url)
    with open("local_model/config.json", "wb") as f:
        f.write(response.content)

    MODEL_PATH = "local_model"

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    return model

# Define the conversion function
def convert_to_2_scale(arr):
    arr_2_scale = []
    for val in arr:
        if val in [0, 1]:
            arr_2_scale.append(0)  # bad
        else:
            arr_2_scale.append(1)  # average
    return np.array(arr_2_scale)

# Load the cached model
# model = load_model()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Streamlit app
st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80", caption="Wine", use_column_width=True)
st.title('Binary Wine Sentiment Analysis')

user_input = st.text_area("Enter the review of the wine:")
if st.button('Predict'):
    # inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    # URLs to your model and config
    url=f'https://vino-api-v2-766cav374q-an.a.run.app/predict?review={user_input}'
    response = requests.get(url)
    verdict = response.json()['verdict']
    if verdict == 'good':
        st.markdown(f"<h1 style='text-align: center; color: green;'>This wine is: {verdict.upper()}</h1>", unsafe_allow_html=True)
        st.image("images/great_wine_wave.png", use_column_width=True)
    else:
        st.markdown(f"<h1 style='text-align: center; color: red;'>This wine is: {verdict.upper()}</h1>", unsafe_allow_html=True)
        st.image("images/bad_wine.png", use_column_width=True)
