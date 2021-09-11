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
response = requests.get('https://raw.githubusercontent.com/famutimine/covid-prism/main/covid.jpg')
image = Image.open(BytesIO(response.content))
st.image(image)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown('''_A Real Time **COVID**-19 **P**ersonalized **R**isk **I**ntelligence **S**ystem for **M**ortality (COVID-PRISM)_.''')
st.markdown('''**Important Note**: COVID-PRISM was developed using data from 1,202 patients hospitalized with a diagnosis of COVID-19 at the University of Missouri Healthcare-Columbia during April 1, 2020 through April 30, 2021.
This artificial intelligence-based prognostic tool has been internally validated to predict 24-48 hour mortality risk and has an area under the receiver operating characteristic curve (AUROC) of 0.93, sensitivity of 95% and specificity of 75%.''')
covid_df=pd.read_csv('https://raw.githubusercontent.com/famutimine/covid-prism/main/covid_dataset.csv',index_col=[0])
covid_df=covid_df.drop(['HGB','Creatinine'],axis=1)
X = covid_df.iloc[:, :-1]
Y = covid_df.iloc[:, -1:]
model=XGBClassifier()
model.fit(X, Y)
st.header('Please fill in current values')
def user_input_features():
    input_features = {}
    input_features["BUN"] = st.number_input(label='Blood Urea Nitrogen (mg/dL)', value=33)
    input_features["CRP"] = st.number_input(label='C-reactive Protein (mg/L)', value=32.33, format="%.2f")
    input_features["RR"] = st.number_input(label='Respiratory Rate (breaths/min)', value=16)
    input_features["HR"] = st.number_input(label='Heart Rate (beats/min)', value=101)
    input_features["SBP"] = st.number_input(label='Systolic Blood Pressure (mmHg)', value=132)
    input_features["Albumin"] = st.number_input(label='Serum Albumin (g/L)', value=2.30, format="%.2f")
    input_features["Lymphocyte count"] = st.number_input(label='Absolute Lymphocyte Count (10^9/L)', value=0.30, format="%.2f")
    input_features["SpO2"] = st.number_input(label='Blood Oxygen Saturation (%)', value=89)
    return [input_features]

df = user_input_features()
feature_names= ['BUN', 'CRP', 'RR', 'HR', 'SBP', 'Albumin', 'Lymphocyte count', 'SpO2'] 
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
    st.write("In-Hospital Mortality Probability: ", str(round(float(probability),2)*100) +"%")
    st.write('---')

    p,shap_values = explain_model_prediction(df.iloc[0])
    st.subheader('Model Prediction Interpretation Plot')
    st_shap(p)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df.iloc[0])
    # st.write(explainer.expected_value + shap_values[0].sum()) # Model Output Calculation
    st.write('''Variables in red increased the prediction while variables in blue decreased prediction for this patient. Please hover on unlabelled arrow bands for visibility''')
        
    st.subheader('Variable Impact on Model Prediction')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values, df)
    st.pyplot(fig)
    st.write('''Variables on the right of the vertical line (at x=0) have positive impact on prediction (increases probability) while variables on the left have negative impact''')
    
    st.subheader('Variable Importance Plot')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values, df, plot_type='bar', max_display=10)
    st.pyplot(fig)
    
    st.write('---')
    st.markdown('''**Disclaimer**: This tool (hereinafter referred to as "COVID-PRISM / Algorithm") is being made publicly available for academic and research purposes only and is not intended for the diagnosis or treatment of any disease or condition, including COVID-19 in individual patients. COVID-PRISM is not a substitute for independent clinical assessment or judgement. All representations and warranties regarding the Algorithm, including warranties of fitness for use in clinical decision making and warranties that the Algorithm works as intended, is clinically safe, does not infringe on third party intellectual property rights, and/or is free from defects and bugs, are hereby disclaimed.''')
st.write('---')
st.markdown('''**Developer/Author Information**''')
st.markdown('''Olubusayo Daniel Famutimi MBBS, MPH''')
st.markdown('''email: _famutimio@health.missouri.edu_''')
hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)
