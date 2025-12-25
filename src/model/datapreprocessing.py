import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

class Preprocessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        return self.df

    def encode_features(self):
        cat_cols = self.df.select_dtypes(include="object").columns

        for col in cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])

        return self.df

    def save_data(self, filename="iris_preprocessed.csv"):
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, filename)
        self.df.to_csv(output_file, index=False)
        return output_file
