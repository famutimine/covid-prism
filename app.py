import streamlit as st
import requests
import datetime
import shap
import json
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import xgboost
from xgboost import XGBClassifier
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
import flask
from flask import Flask, request, redirect, url_for, flash, jsonify, make_response
from PIL import Image

st.set_page_config(layout="centered")
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
image = Image.open('https://raw.githubusercontent.com/famutimine/covid-prism/main/covid.jpg')
st.image(image)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown('''_A Real Time **COVID**-19 **P**ersonalized **R**isk **I**ntelligence **S**ystem for **M**ortality_.''')
st.markdown('''**Important Note**: The current model (COVID-PRISM) was developed using data from 1,202 patients hospitalized
with a diagnosis of COVID-19 at the University of Missouri Healthcare-Columbia during April 1, 2020 through April 30, 2021.
COVID-PRISM has been internally validated and has an area under the receiver operating characteristic curve (AUROC) of 0.93.
A probability threshold value of 12% has been determined to be highly predictive of mortality in the study population.
This threshold was calculated based on the value that optimized sensitivity and specificity of the model.
This means that any patient with a mortality probability of 12% or higher is considered high risk for mortality within the next 24-48 hours.''')

covid_df=pd.read_csv('https://raw.githubusercontent.com/famutimine/covid-prism/main/covid_data.csv',index_col=[0])
X = covid_df.iloc[:, :-1]
Y = covid_df.iloc[:, -1]
st.header('Please fill in current values')
def user_input_features():
    input_features = {}
    input_features["BUN"] = st.number_input(label='Blood Urea Nitrogen (mg/dL)', value=35)
    input_features["CRP"] = st.number_input(label='C-reactive Protein (mg/L)', value=40.00, format="%.2f")
    input_features["RR"] = st.number_input(label='Respiratory Rate (breaths/min)', value=35)
    input_features["Creatinine"] = st.number_input(label='Serum Creatinine (mg/L)', value=0.50, format="%.2f")
    input_features["HR"] = st.number_input(label='Heart Rate (beats/min)', value=130)
    input_features["SBP"] = st.number_input(label='Systolic Blood Pressure (mmHg)', value=90)
    input_features["Albumin"] = st.number_input(label='Serum Albumin  (g/L)', value=2.30, format="%.2f")
    input_features["HGB"] = st.number_input(label='Hemoglobin level (g/dL)', value=15.00, format="%.2f")
    input_features["Lymphocyte count"] = st.number_input(label='Absolute Lymphocyte Count (10^9/L)', value=0.30, format="%.2f")
    input_features["SpO2"] = st.number_input(label='Blood Oxygen Saturation (%)', value=92)
    return [input_features]


model=XGBClassifier()
model.fit(X, Y)
df = user_input_features()
feature_names= ['BUN', 'CRP', 'RR', 'Creatinine', 'HR', 'SBP', 'Albumin', 'HGB', 'Lymphocyte count', 'SpO2'] 
df = pd.DataFrame(df,columns = feature_names)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
def explain_model_prediction(data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    p = shap.force_plot(explainer.expected_value, shap_values, data)
    return p, shap_values



submit = st.button('Get predictions')
if submit:
    probability = model.predict_proba(df)[:,1]
    st.header('Model Prediction')
    st.write("In-Hospital Mortality Probability: ", str(round(float(probability),3)*100) +"%")
    st.write('---')

    p,shap_values = explain_model_prediction(df)
    st.subheader('Model Prediction Interpretation Plot')
    st_shap(p)
    
    st.subheader('Summary Plot 1')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values, df)
    st.pyplot(fig)
    st.subheader('Summary Plot 2')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values, df, plot_type='bar', max_display=10)
    st.pyplot(fig)
    st.markdown('''**Disclaimer**: This tool (hereinafter referred to as "COVID-PRISM / Algorithm") is being made publicly available for academic and research purposes only and is not intended for the diagnosis or treatment of any disease or condition, including COVID-19 in individual patients. COVID-PRISM is not a substitute for independent clinical assessment or judgement. All representations and warranties regarding the Algorithm, including warranties of fitness for use in clinical decision making and warranties that the Algorithm works as intended, is clinically safe, does not infringe on third party intellectual property rights, and/or is free from defects and bugs, are hereby disclaimed.''')
st.write('---')
st.markdown('''**Author Information**''')
st.markdown('''Olubusayo Daniel Famutimi MD, MPH''')
st.markdown('''email: _famutimio@health.missouri.edu_''')
hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)
