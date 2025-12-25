import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os

class Trainer:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path

    def load_data(self):
        X_train = pd.read_csv(os.path.join(self.data_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(self.data_path, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(self.data_path, "y_train.csv"))
        y_test = pd.read_csv(os.path.join(self.data_path, "y_test.csv"))
        return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

    def build_pipeline(self):
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000))
            ]
        )

    def train(self, X_train, y_train):
        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("Accuracy:", acc)
        print(classification_report(y_test, preds))
        return acc

    def save_model(self, model, model_name="logistic_model"):
        os.makedirs(self.output_path, exist_ok=True)
        model_file = os.path.join(self.output_path, f"{model_name}.pkl")
        mlflow.sklearn.save_model(model, model_file)
        return model_file

    def train_with_mlflow(self):
        X_train, X_test, y_train, y_test = self.load_data()
        with mlflow.start_run():
            model = self.train(X_train, y_train)
            acc = self.evaluate(model, X_test, y_test)

            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

            print("Model trained and logged to MLflow")
