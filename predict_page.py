import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open("saved_steps.pkl", "rb") as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_sex = data["le_sex"]
le_smoker = data["le_smoker"]
le_region = data["le_region"]

def show_predict_page():
    st.title("Insurance Charge Prediction")
    st.write("------------")


    st.write("""
        ##### Input information to predict the insurance charge
    """)

    sex = (
        "male",
        "female"
    )

    smoker = (
        "yes",
        "no"
    )

    region = (
        "southeast",
        "northwest",
        "northeast",
        "southwest"
    )

    age = st.slider("Age", min_value=18, max_value=64, help="Slide this button to choose your age")

    sex_choice = st.selectbox("Sex", sex, help="Choose whether you are a male of female")

    bmi = st.slider("BMI",min_value=15.000, max_value=50.000, step=0.01, help="Slide this button to choose you BMI")

    smoker_choice = st.selectbox("Smoking?", smoker, help="Choose whether you smoke or not" )

    children = st.slider("Number of children", min_value=0, max_value=7, help="Choose how many children you have")

    region_choice = st.selectbox("Region", region, help="Choose region where you live")


    ok = st.button("Calculate insurance charge",help="Predict your insurance charge based on the provided data above")

    if ok:
        array = np.array([[age, sex_choice, bmi, children, smoker_choice, region_choice]])
        array[:, 1] = le_sex.transform(array[:, 1])
        array[:, 4] = le_smoker.transform(array[:, 4])
        array[:, -1] = le_region.transform(array[:, -1])
        index_values = [0]
        column_values = ["age", "sex", "bmi", "children", "smoker", "region"]
        X = pd.DataFrame(data = array, 
                        index = index_values,
                        columns = column_values)

        charge = regressor_loaded.predict(X)
        st.write("------------")
        st.subheader(f"The estimated charge is ${charge[0]:.2f}")
