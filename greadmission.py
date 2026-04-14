import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Admission Predictor")

st.title("Graduate Admission Predictor")

data = pd.read_csv("Admission_Predict.csv")

data = data.drop(columns=["Serial No."])

X = data.drop("Chance of Admit ", axis=1)
y = data["Chance of Admit "]

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

if "ready" not in st.session_state:
    st.session_state.ready = False

if st.button("Train Model"):
    model = train_model()
    st.session_state.model = model
    st.session_state.ready = True
    st.success("Model trained")

if st.session_state.ready:

    gre = st.number_input("GRE Score", min_value=260, max_value=340, value=300)
    toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
    rating = st.number_input("University Rating", min_value=1, max_value=5, value=3)
    sop = st.slider("SOP Strength", 1.0, 5.0, 3.0)
    lor = st.slider("LOR Strength", 1.0, 5.0, 3.0)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0)
    research = st.selectbox("Research Experience", [0, 1])

    if st.button("Predict"):

        input_data = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])

        prediction = st.session_state.model.predict(input_data)[0]

        st.subheader(f"Chance of Admission: {round(prediction * 100, 2)}%")

else:
    st.info("Train the model first")