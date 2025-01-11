import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class HeartFailurePrediction:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.df = self.df.dropna()

    def preprocess(self):
        categorical_columns = ["sex", "cp", "thal"]
        self.df = pd.get_dummies(self.df, columns=categorical_columns)

        scaler = StandardScaler()
        numerical_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.df[numerical_columns] = scaler.fit_transform(self.df[numerical_columns])

    def train_model(self):
        X = self.df.drop("target", axis=1)
        y = self.df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        print("Random Forest Model Accuracy:", accuracy_score(y_test, y_pred))
        return X_test, y_pred
