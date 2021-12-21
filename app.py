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
from io import BytesIO

st.set_page_config(layout="centered")
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
#st.markdown("<h1 style='text-align: center; color: red;'>COVID-PRISM</h1>", unsafe_allow_html=True)
response = requests.get('https://raw.githubusercontent.com/famutimine/covid-prism/main/covid.jpg')
image = Image.open(BytesIO(response.content))
st.image(image)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown('''_A Real Time **COVID**-19 **P**ersonalized **R**isk **I**ntelligence **S**ystem for **M**ortality (COVID-PRISM)_.''')
st.markdown('''**Important Note**: COVID-PRISM is artificial intelligence-based prognostic model developed using data from 1,917 patients hospitalized with a diagnosis of COVID-19 at the University of Missouri Healthcare-Columbia during April 1, 2020 through November 30, 2021.
This model has been internally validated to predict 24-48 hour inpatient mortality risk with an area under the receiver operating characteristic curve (AUROC) of 0.97, sensitivity of 89% and specificity of 94%.''')
st.write('---')
st.markdown('''**Disclaimer**: This tool (hereinafter referred to as "COVID-PRISM / Algorithm") is being made publicly available for academic and research purposes only and is not intended for the diagnosis or treatment of any disease or condition, including COVID-19 in individual patients. COVID-PRISM is not a substitute for independent clinical assessment or judgement. All representations and warranties regarding the Algorithm, including warranties of fitness for use in clinical decision making and warranties that the Algorithm works as intended, is clinically safe, does not infringe on third party intellectual property rights, and/or is free from defects and bugs, are hereby disclaimed.''')
covid_df=pd.read_csv('https://raw.githubusercontent.com/famutimine/covid-prism/main/covid19_data.csv')
X = covid_df.iloc[:, :-1]
Y = covid_df.iloc[:, -1:]
model=XGBClassifier()
model.fit(X, Y)
st.header('Enter the most recent values within the last 24 hours')
def user_input_features():
    input_features = {}
    input_features["Albumin"] = st.number_input(label='Serum Albumin (g/L)', value=3.20, format="%.2f")
    input_features["BUN"] = st.number_input(label='Blood Urea Nitrogen (mg/dL)', value=23.00, format="%.2f") 
    input_features["SpO2_FiO2_Ratio"] = st.number_input(label='SpO2:FiO2 Ratio', value=180)
    input_features["Respiratory_Rate"] = st.number_input(label='Respiratory Rate (breaths/min)', value=42) 
    input_features["HGB"] = st.number_input(label='Hemoglobin Level (g/dL)', value=12.7)
    input_features["Heart_Rate"] = st.number_input(label='Heart Rate (beats/min)', value=118)
    input_features["SBP"] = st.number_input(label='Systolic Blood Pressure (mmHg)', value=164)
    return [input_features]

df = user_input_features()
feature_names= ['Albumin', 'BUN', 'SpO2_FiO2_Ratio', 'Respiratory_Rate',  'HGB','Heart_Rate','SBP'] 
df = pd.DataFrame(df,columns = feature_names)


submit = st.button('Get predictions')
if submit:
    probability = model.predict_proba(df)[:,1]
    st.header('Model Prediction')
    st.write("Risk of Severe Illness or In-Hospital Mortality: ", str(round(float(probability),2)*100) +"%")
    st.write('---')
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df.iloc[0])
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])
    st.pyplot(fig)
    st.write('''Variables in red increased the prediction while variables in blue decreased prediction for this patient. The magnitude of effect of each variable is indicated by the numerical value labels.''')
        
st.write('---')
st.markdown('''**Developer/Author Information**''')
st.markdown('''Olubusayo Daniel Famutimi, MD, MPH''')
st.markdown('''email: _famutimio@health.missouri.edu_''')
st.markdown('''**Contributors & Co-Authors**''')
st.markdown('''Tinashe Maduke, MD, MPH''')
st.markdown('''Hariharan Regunath, MD, FACP, FIDSA''')

hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)
