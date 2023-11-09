from matplotlib import pyplot as plt
import pandas as pd


def show_timeseries(data: pd.Series, dates: pd.Series):
    """
    takes data, prices as arguments
    and shows linegraph
    """

    return 0

def show_correlation(series_1: pd.Series, series_2: pd.Series):
    
    """
    takes DataFrame and two columns (float/int)
    then shows correlation on x, y surface
    """
    
    print(f"Correlation between {series_1.name} and {series_2.name}:\n")
    print(series_1.corr(series_2), '\n')

    x = series_1
    y = series_2

    fig, ax =  plt.subplots(figsize=(9, 7), dpi=100)

    ax.scatter(x, y, alpha=0.3, s=10)

    plt.xlabel(f"{series_1.name}", fontsize=14, labelpad=15)
    plt.ylabel(f"{series_2.name}", fontsize=14, labelpad=15)
    plt.title("bank_clients", fontsize=14, pad=15)
    plt.show()

    return 0