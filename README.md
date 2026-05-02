# Credit Risk Prediction using Machine Learning

## Overview

This project uses Machine Learning techniques to predict whether a loan applicant is likely to default or repay the loan successfully. The model analyzes applicant financial and personal data to classify credit risk as **Low Risk** or **High Risk**.

## Objective

To help banks and financial institutions make smarter lending decisions by reducing default risk through predictive analytics.

## Features

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature selection
* Model training and evaluation
* Credit risk classification
* Performance comparison of multiple ML algorithms

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## Machine Learning Algorithms

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine

## Dataset Attributes

The dataset may include:

* Age
* Income
* Employment Status
* Loan Amount
* Credit History
* Existing Debts
* Marital Status
* Education Level

## Workflow

1. Import dataset
2. Clean missing/null values
3. Encode categorical features
4. Split data into training and testing sets
5. Train ML models
6. Evaluate model performance
7. Predict customer credit risk

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

## Results

Random Forest / Logistic Regression achieved strong performance in identifying risky applicants with reliable accuracy.

## Project Structure

```bash
Credit-Risk-Prediction/
│── dataset.csv
│── credit_risk.ipynb
│── app.py
│── requirements.txt
│── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/credit-risk-prediction.git
cd credit-risk-prediction
pip install -r requirements.txt
```

## Run Project

```bash
python app.py
```
