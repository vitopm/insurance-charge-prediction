from matplotlib.pyplot import show
import streamlit as st
from predict_page import show_predict_page

st.sidebar.selectbox("Explore or Predict", ("Predict", "Explore Page"))

show_predict_page()