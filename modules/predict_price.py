import process

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.functional as F

def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    """
    takes DataFrame with column 'date' and change dates 
    to the number of days from the earliest day
    """

    df['date'] = pd.to_datetime(df['date'], yearfirst=True, format=r'%d.%m.%Y')

    df['date'] = (df['date'] - df['date'].min()).dt.days

    df.rename({'date': 'day'}, axis=1, inplace=True)

    return df

def main():

    df = pd.read_csv('datasets/Predict_Future_Sales_sales_train.csv')

    df = preprocess(df)

    print(df.head())

if __name__ == "__main__":
    main()
