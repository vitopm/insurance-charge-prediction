import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page


st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select page", ("Predict", "Explore Data"))
# pagePredict = st.sidebar.button("Predict")
# pageExplore = st.sidebar.button("Explore Page")
if page == "Predict":
    show_predict_page()
else:
    show_explore_page()