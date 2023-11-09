import process
import pandas as pd
import matplotlib.pyplot as plt


def main():
    
    df = pd.read_csv('datasets/Churn_for_Bank_Customers_Kaggle.csv')

    process.show_correlation(df["CreditScore"], df["Balance"])


if __name__ == "__main__":
    main()