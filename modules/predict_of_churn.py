import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import argparse
import random
import json
import os


device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") Doesn't work right now
print("\nStarting with device:", device)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed) cuda seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}\n")


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
        hidden_features   = 36
        hidden_features_2 = 18

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_features),
            nn.Dropout(p=0.25),

            nn.Linear(hidden_features, hidden_features_2),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.1),

            nn.Linear(hidden_features_2, 1)
        )

    def forward(self, x):
        y_pred = F.sigmoid(self.linear_layer(x))
        return y_pred


class Worker:

    def __init__(self, settings: dict):
        self.settings = settings

        self.dataprocessor = DataProcessor(settings["file_name"])

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(self.dataprocessor.X, self.dataprocessor.y, test_size=0.2)
        )

        self.scaler = StandardScaler()

        print("Dataset is ready now")
        print(self.X_train.head())

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.fit_transform(self.X_test)

        self.X_train = torch.from_numpy(self.X_train.astype(np.float32)).to(device)
        self.X_test = torch.from_numpy(self.X_test.astype(np.float32)).to(device)
        self.y_train = torch.from_numpy(self.y_train.to_numpy().astype(np.float32)).to(device).view(-1, 1)
        self.y_test = torch.from_numpy(self.y_test.to_numpy().astype(np.float32)).to(device).view(-1, 1)

        self.dataset = TensorDataset(self.X_train, self.y_train)
        self.nn = NeuralNetwork(in_features=self.X_train.shape[1]).to(device)

        self.train_acc_history = []
        self.test_acc_history = []
        self.loss_history = []

    def train_model(self):
        print("\nStarting Training")

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.settings["learning_rate"])

        loader = DataLoader(self.dataset, batch_size=self.settings["batch_size"], shuffle=True)

        # Training loop
        for epoch in tqdm(range(self.settings["num_epochs"])):
            self.nn.train()
            loss = 0

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
                # acc = (y_pred.round() == y).float().mean()

            # Debug
            self.nn.eval()

            self.train_acc_history.append(self.test_model())
            self.test_acc_history.append(self.eval_model())
            self.loss_history.append(loss.item())

            # Printing provisional results
            if epoch % 10 == 9 or epoch == 0:
                print(f"Epoch {epoch + 1}: Loss {loss.item():.4f}: Accuracy {self.test_acc_history[-1]}")

        print("Finished Training")

    def show_stats(self, savefig=None):
        plt.plot(self.train_acc_history, label="Train")
        plt.plot(self.test_acc_history, label="Test")

        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.grid()

        if savefig:
            plt.savefig(savefig)
        else:
            plt.show()

    def test_model(self):
        with torch.no_grad():
            y_pred = self.nn(self.X_train)
            acc = (y_pred.round() == self.y_train).float().mean()
            return float(acc)

    def eval_model(self):
        with torch.no_grad():
            y_pred = self.nn(self.X_test)
            acc = (y_pred.round() == self.y_test).float().mean()
            return float(acc)

    def save_model(self):
        directory = "predict_of_churn/model"
        print(f"\nSaving at directory: {directory}")

        torch.save(self.nn.state_dict(), f"{directory}/state_dict")

        json.dump(self.train_acc_history, open(f"{directory}/train_acc_history.json", mode="w"), indent=2)
        json.dump(self.test_acc_history, open(f"{directory}/test_acc_history.json", mode="w"), indent=2)

        self.show_stats(savefig=f"{directory}/train_history.png")

    def load_model(self):
        directory = "predict_of_churn/model"
        print(f"\nLoading from directory: {directory}")

        self.train_acc_history = json.load(open(f"{directory}/train_acc_history.json", mode="r"))
        self.test_acc_history = json.load(open(f"{directory}/test_acc_history.json", mode="r"))

        self.nn.load_state_dict(torch.load(f"{directory}/state_dict"))
        self.nn.eval()


def main(mode: str = "demonstrate"):
    configuration = json.load(open("predict_of_churn/settings.json", mode="r"))
    set_seed(configuration["seed"])

    worker = Worker(settings=configuration)

    if mode == "demonstrate":
        worker.load_model()
        worker.show_stats()
    elif mode == "rewrite":
        worker.train_model()
        worker.show_stats()
        worker.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Neural Network for predicting of churn!")
    parser.add_argument(
        "--mode",
        type=str,
        default="demonstrate",
        help="select working mode (default: demonstrate)"
    )
    run_args = parser.parse_args()

    main(mode=run_args.mode)
