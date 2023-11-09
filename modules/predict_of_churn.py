import pandas as pd


def main():
    
    df = pd.read_csv('datasets/Churn_for_Bank_Customers_Kaggle.csv')

    print(df.head())
