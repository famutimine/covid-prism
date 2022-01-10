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
st.markdown('''**Background**: COVID-PRISM is artificial intelligence-based prognostic model developed at the University of Missouri Healthcare-Columbia using a cohort of 1,917 patients hospitalized with a diagnosis of COVID-19 during April 1, 2020 through November 30, 2021.
This model has been internally validated to predict 24-hour and 7-day risk of progression to severe illness or inpatient mortality. Model achieved area under the receiver operating characteristic curve (AUROC) score of 0.974, sensitivity of 90% and specificity of 92.8% for predicting 24-hour risk, and AUROC score of 0.953, sensitivity of 87.6% and specificity of 92% for predicting 7-day risk.''')
covid_df=pd.read_csv('https://raw.githubusercontent.com/famutimine/covid-prism/main/covid19_data.csv')
covid_df_7=pd.read_csv('https://raw.githubusercontent.com/famutimine/covid-prism/main/covid19_data_7.csv')
covid_df.rename(columns={"SpO2_FiO2_Ratio":"SpO2:FiO2 Ratio","BUN":"Blood Urea Nitrogen","Respiratory_Rate":"Respiratory Rate","HGB":"Hemoglobin","Heart_Rate":"Heart Rate","SBP":"Systolic Blood Pressure"},inplace = True)
covid_df_7.rename(columns={"SpO2_FiO2_Ratio":"SpO2:FiO2 Ratio","BUN":"Blood Urea Nitrogen","Respiratory_Rate":"Respiratory Rate","HGB":"Hemoglobin","Heart_Rate":"Heart Rate","SBP":"Systolic Blood Pressure"},inplace = True)
X = covid_df.iloc[:, :-1]
Y = covid_df.iloc[:, -1:]
X_7 = covid_df_7.iloc[:, :-1]
Y_7 = covid_df_7.iloc[:, -1:]
model=XGBClassifier()
model.fit(X, Y)
model7=XGBClassifier()
model7.fit(X_7, Y_7)
st.write('---')
st.markdown('**For vital sign variables (including SpO2:FiO2 Ratio), enter the most recent value within the last 24 hours. For laboratory variables, enter the most recent value in the last 72 hours.**')
st.markdown('**NB:** For missing values, please leave blank. Missing values will be automatically imputed using Multivariate Imputation by Chained Equations')
def user_input_features():
    input_features = {}
    input_features["Albumin"] = st.text_input(label='Serum Albumin (g/L)', value="", help="Leave blank if value is missing")
    input_features["Blood Urea Nitrogen"] = st.text_input(label='Blood Urea Nitrogen (mg/dL)', value="", help="Leave blank if value is missing") 
    input_features["SpO2:FiO2 Ratio"] = st.number_input(label='SpO2:FiO2 Ratio', value=180, help="This field is required")
    input_features["Respiratory Rate"] = st.number_input(label='Respiratory Rate (breaths/min)', value=42, help="This field is required") 
    input_features["Hemoglobin"] = st.text_input(label='Hemoglobin Level (g/dL)', value="", help="Leave blank if value is missing")
    input_features["Heart Rate"] = st.number_input(label='Heart Rate (beats/min)', value=118, help="This field is required")
    input_features["Systolic Blood Pressure"] = st.number_input(label='Systolic Blood Pressure (mmHg)', value=164, help="This field is required")
    return [input_features]
    
        
df = user_input_features()
feature_names= ['Albumin', 'Blood Urea Nitrogen', 'SpO2:FiO2 Ratio', 'Respiratory Rate',  'Hemoglobin','Heart Rate','Systolic Blood Pressure'] 
df = pd.DataFrame(df,columns = feature_names)
try:
   df[["Albumin", "Blood Urea Nitrogen", "Hemoglobin"]] = df[["Albumin","Blood Urea Nitrogen", "Hemoglobin"]].apply(pd.to_numeric) 
except ValueError:
    st.error('Please enter a valid input')
    st.stop()
X=pd.concat([X, df])
lreg = LinearRegression()
imp = IterativeImputer(estimator=lreg,missing_values=np.nan, max_iter=40, verbose=2, imputation_order='roman',random_state=123,tol=0.00001,min_value=0)
X=imp.fit_transform(X)
X = pd.DataFrame(X, columns=feature_names)
cols=["Albumin", "Blood Urea Nitrogen", "Hemoglobin"]
for col in cols:
    X[col]=X[col].round(1)


df=X.tail(1)
X_7=pd.concat([X_7, df])
for col in cols:
    X_7[col]=X_7[col].round(1)

submit = st.button('Get predictions')
if submit:
    probability = model.predict_proba(df)[:,1]    
    st.header('Model Prediction for 24-hour Risk of Progression to Severe Illness or Mortality')
    st.write("24-hour Risk of Progression to Severe Illness or Mortality: ", str(round(float(probability*100),1)) +"%")
    st.write('---')
    
    st.subheader('SHAP Waterfall Plot for Model Explanation and Interpretation (24-Hour Risk)')
    fig, ax = plt.subplots()    
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    idx=len(X)-1
    shap.plots.waterfall(shap_values[idx])    
    st.pyplot(fig)
    st.write('''Variables corresponding to the red arrow increased the prediction (increased the risk), while variables corresponding to the blue arrow decreased prediction (decreased the risk) for this patient. The magnitude of effect of each variable is indicated by the numerical value labels.''')
    
    probability7 = model7.predict_proba(df)[:,1]
    st.header('Model Prediction for 7-day Risk of Progression to Severe Illness or Mortality')
    st.write("7-day Risk of Progression to Severe Illness or Mortality: ", str(round(float(probability7*100),1)) +"%")
    st.write('---')
    
    st.subheader('SHAP Waterfall Plot for Model Explanation and Interpretation (7-day Risk)')
    fig7, ax7 = plt.subplots()
    explainer7 = shap.Explainer(model7)
    shap_values = explainer7(X_7)
    idx7=len(X_7)-1
    shap.plots.waterfall(shap_values[idx7])
    st.pyplot(fig7)
    st.write('''Variables corresponding to the red arrow increased the prediction (increased the risk), while variables corresponding to the blue arrow decreased prediction (decreased the risk) for this patient. The magnitude of effect of each variable is indicated by the numerical value labels.''')
    
st.write('---')
st.markdown('''**Disclaimer**: This tool (hereinafter referred to as "COVID-PRISM / Algorithm") is being made publicly available for academic and research purposes only and is not intended for the diagnosis or treatment of any disease or condition, including COVID-19 in individual patients. COVID-PRISM is not a substitute for independent clinical assessment or judgement. All representations and warranties regarding the Algorithm, including warranties of fitness for use in clinical decision making and warranties that the Algorithm works as intended, is clinically safe, does not infringe on third party intellectual property rights, and/or is free from defects and bugs, are hereby disclaimed.''')

hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)
