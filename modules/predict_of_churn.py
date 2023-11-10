import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import process


settings = {
    "file_name": "datasets/Churn_for_Bank_Customers_Kaggle.csv"
}


class DataProcessor:
    def __init__(self, file_name: str = "datasets/Churn_for_Bank_Customers_Kaggle.csv"):
        print(self.preprocess_df(self.get_df(file_name)))
        self.df = self.preprocess_df(self.get_df(file_name))
        self.X, self.y = self.XY_split()


    @staticmethod
    def get_df(file_name: str = "datasets/Churn_for_Bank_Customers_Kaggle.csv") -> pd.DataFrame:
        return pd.read_csv(file_name)


    @staticmethod
    def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(labels=["RowNumber", "CustomerId", "Surname"], axis=1)

        new_df = []

        country_to_list = {
            "France" : [1, 0, 0],
            "Spain"  : [0, 1, 0],
            "Germany": [0, 0, 1],
        }

        sex_to_list = {
            "Male": [1],
            "Female": [0]
        }

        for i in range(len(df)):
            value = df.iloc[i]["Geography"]
            r = country_to_list[value]


            value = df.iloc[i]["Gender"]
            r = r + sex_to_list[value]
            new_df.append(r)

        new_df = pd.DataFrame(new_df, columns=["France", "Spain", "Germany", "Male"])
        print(new_df)
        print(df.shape, new_df.shape)
        print(df)
        df = df + new_df
        print(df)

        return df.drop(labels=["Geography", "Gender"], axis=1)


    def XY_split(self):
        print(self.df.head())
        return self.df.drop(labels=["Excited"], axis=1), self.df["Excited"]


class Main:

    def __init__(self, settings: dict):
        self.settings = settings

        self.dataprocessor = DataProcessor(settings["file_name"])

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(self.dataprocessor.X, self.dataprocessor.y, test_size=0.2)
        )

        self.scaler = StandardScaler()

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test  = self.scaler.fit_transform(self.X_test)


def main():
    pass
    # df = pd.read_csv('datasets/Churn_for_Bank_Customers_Kaggle.csv')
    #
    # process.show_correlation(df["CreditScore"], df["Balance"])


if __name__ == "__main__":
    # main()
    main = Main(settings=settings)
