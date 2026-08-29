# Diabetes-Prediction-ML
Diabetes Prediction using Machine Learning and Streamlit
# 🩺 Diabetes Prediction using Machine Learning

## 📌 Project Overview

This mini project uses machine learning to predict the likelihood of diabetes based on patient health-related features.

The project compares Logistic Regression and Random Forest classification algorithms and uses hyperparameter tuning to improve the Random Forest model.

A Streamlit web application is developed to provide an interactive user interface for making predictions.

## 🎯 Problem Statement

Diabetes is a common health condition that can be difficult to identify at an early stage. This project aims to build a machine learning classification system that predicts whether a given set of patient measurements belongs to the diabetic or non-diabetic class.

## 📊 Dataset

The project uses a diabetes dataset containing the following features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Target Variable

- `0` — Non-diabetic class
- `1` — Diabetic class

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- GitHub

## 🤖 Machine Learning Algorithms

### 1. Logistic Regression

Used as the baseline classification model.

### 2. Random Forest

Used as the second classification model.

### 3. Hyperparameter Tuning

GridSearchCV was used to find better Random Forest hyperparameters.

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Random Forest
   ↓
Model Evaluation
   ↓
Hyperparameter Tuning
   ↓
Final Model
   ↓
Streamlit Application
