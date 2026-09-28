# Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster.

## Project Overview

This project uses the [Titanic dataset from Kaggle](https://www.kaggle.com/competitions/titanic) to build a machine learning model for survival prediction.

The project covers data analysis, data cleaning, visualization, feature preprocessing, model training, and model evaluation.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

titanic-survival-prediction/
│
├── data/
│   └── train.csv
├── analysis.ipynb
├── README.md
├── .gitignore
└── requirements.txt
```

## Dataset

Dataset: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)

The dataset contains information about Titanic passengers, including:

- Passenger class
- Gender
- Age
- Fare
- Number of siblings/spouses
- Number of parents/children
- Survival status

## Workflow

1. Load the dataset
2. Explore the data
3. Handle missing values
4. Analyze and visualize the data
5. Encode categorical features
6. Train a Random Forest classifier
7. Evaluate model performance

## Machine Learning

The project uses Scikit-learn to train and evaluate a classification model.

The target variable is:

- `Survived = 1` → Passenger survived
- `Survived = 0` → Passenger did not survive

## Results

- Model: Random Forest Classifier
- Accuracy: 82%
- Precision / Recall (survived): 0.80 / 0.76

## Goal

Build a machine learning model that can predict whether a Titanic passenger survived based on passenger information.

## Author

Deepak Pradhan