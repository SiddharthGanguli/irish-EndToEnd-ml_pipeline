import pandas as pd
from sklearn.model_selection import train_test_split
import os

class TrainTestSplit:
    def __init__(self, input_path, output_path, target_column):
        self.input_path = input_path
        self.output_path = output_path
        self.target_column = target_column
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        return self.df

    def split(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def save_splits(self, X_train, X_test, y_train, y_test):
        os.makedirs(self.output_path, exist_ok=True)

        X_train_path = os.path.join(self.output_path, "X_train.csv")
        X_test_path = os.path.join(self.output_path, "X_test.csv")
        y_train_path = os.path.join(self.output_path, "y_train.csv")
        y_test_path = os.path.join(self.output_path, "y_test.csv")

        X_train.to_csv(X_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)

        return X_train_path, X_test_path, y_train_path, y_test_path
