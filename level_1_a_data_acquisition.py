import pandas as pd


def read_csv(*args, **kwargs):

    df = pd.read_csv(*args, **kwargs)

    return df


