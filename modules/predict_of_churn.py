import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import process


settings = {
    "file_name": "datasets/Churn_for_Bank_Customers_Kaggle.csv",
    "learning_rate": 0.001,
    "num_epochs": 15,
}


class DataProcessor:
    def __init__(self, file_name: str = "datasets/Churn_for_Bank_Customers_Kaggle.csv"):
        self.df = self.preprocess_df(self.get_df(file_name))
        self.X, self.y = self.XY_split()


    @staticmethod
    def get_df(file_name: str = "datasets/Churn_for_Bank_Customers_Kaggle.csv") -> pd.DataFrame:
        return pd.read_csv(file_name)


    @staticmethod
    def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(labels=["RowNumber", "CustomerId", "Surname"], axis=1)

        ohe = OneHotEncoder()
        tf = ohe.fit_transform(df[["Geography"]])
        df[ohe.categories_[0]] = tf.toarray()

        ohe = OneHotEncoder(drop=["Female"])
        tf = ohe.fit_transform(df[["Gender"]])
        df["Male"] = tf.toarray()

        return df.drop(labels=["Geography", "Gender"], axis=1)


    def XY_split(self):
        return self.df.drop(labels=["Exited"], axis=1), self.df["Exited"]


class NeuralNetwork(nn.Module):
    def __init__(self, in_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_features = 36

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

        self.linear_1 = nn.Linear(in_features, hidden_features)
        self.linear_2 = nn.Linear(hidden_features, 1)

    def forward(self, x):
        out = self.dropout(self.relu(self.linear_1(x)))
        y_pred = torch.sigmoid(self.linear_2(out))
        return y_pred


class Main:

    def __init__(self, settings: dict):
        self.settings = settings

        self.dataprocessor = DataProcessor(settings["file_name"])

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(self.dataprocessor.X, self.dataprocessor.y, test_size=0.2)
        )

        self.scaler = StandardScaler()

        print(self.X_train)

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test  = self.scaler.fit_transform(self.X_test)

        self.X_train = torch.from_numpy(self.X_train.astype(np.float32))
        self.X_test  = torch.from_numpy(self.X_test.astype(np.float32))
        self.y_train = torch.from_numpy(self.y_train.to_numpy().astype(np.float32)).view(-1, 1)
        self.y_test  = torch.from_numpy(self.y_test.to_numpy().astype(np.float32)).view(-1, 1)

        self.dataset = TensorDataset(self.X_train, self.y_train)
        self.nn = NeuralNetwork(in_features=self.X_train.shape[1])

        self.acc_history = []
        self.loss_history = []

    def train_model(self):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=settings["learning_rate"])

        loader = DataLoader(self.dataset, batch_size=1000, shuffle=True)

        # Training loop
        for epoch in range(settings["num_epochs"]):
            for X, y in loader:
                # Forward
                y_pred = self.nn(X)

                # Loss computation
                loss = criterion(y_pred, y)

                # Backward
                loss.backward()

                # Update weights
                optimizer.step()

                # Zero gradients
                optimizer.zero_grad()

                # log
                acc = (y_pred.round() == y).float().mean()
                self.acc_history.append(acc)



            # Debug
            if not epoch % 1:
                self.loss_history.append(loss.item())
                print(f"Epoch {epoch+1}: Loss {loss.item():.4f}: Accuracy {self.test_model()}")

        plt.plot(self.acc_history)
        plt.show()

    def test_model(self):
        with torch.no_grad():
            y_pred = self.nn(self.X_test)
            acc = (y_pred.round() == self.y_test).float().mean()
            return acc



def main():
    pass
    # df = pd.read_csv('datasets/Churn_for_Bank_Customers_Kaggle.csv')
    #
    # process.show_correlation(df["CreditScore"], df["Balance"])


if __name__ == "__main__":
    # main()
    main = Main(settings=settings)
    main.train_model()
