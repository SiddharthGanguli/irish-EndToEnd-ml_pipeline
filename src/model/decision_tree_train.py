import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
import mlflow
import mlflow.sklearn
import os
from sklearn.pipeline import Pipeline

class DecisionTreeModel:

    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path

    def load_data(self):
        X_train = pd.read_csv(os.path.join(self.data_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(self.data_path, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(self.data_path, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(self.data_path, "y_test.csv")).values.ravel()
        return X_train, X_test, y_train, y_test

    def build_model(self):
        return DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )

    def train_with_mlflow(self):
        X_train, X_test, y_train, y_test = self.load_data()

        with mlflow.start_run():
            model = self.build_model()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_param("model", "DecisionTree")
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

            print("Accuracy:", acc)
            print(classification_report(y_test, preds))
