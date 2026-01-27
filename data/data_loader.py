from sklearn.datasets import load_breast_cancer
import pandas as pd


def data_loader(as_dataframe = True):
    data = load_breast_cancer()

    if as_dataframe:
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name = 'target')

    else:
        X = data.data
        y = data.target

    return X, y, data.feature_names, data.target_names
