"""
Titanic Survival Model Training Script
Trains a RandomForestClassifier pipeline with preprocessing and exports model.joblib.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_and_save_model(data_path="data/train.csv", model_path="model.joblib"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature columns
    numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Sex', 'Embarked']
    
    X = df[numeric_features + categorical_features]
    y = df['Survived']
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=7))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training model pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("=== Test Set Evaluation ===")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Retrain on full dataset for production artifact
    print("Fitting model on full dataset...")
    model_pipeline.fit(X, y)
    
    # Metadata bundle
    artifact = {
        'model': model_pipeline,
        'features': {
            'numeric': numeric_features,
            'categorical': categorical_features
        },
        'metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
    }
    
    joblib.dump(artifact, model_path)
    print(f"Successfully saved model artifact to {model_path}!")

if __name__ == "__main__":
    train_and_save_model()
