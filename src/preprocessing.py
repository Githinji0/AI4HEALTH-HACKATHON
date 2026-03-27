import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    X, y = [], []

    for features, label in data:
        X.append(features)
        y.append(label)

    X = np.array(X) / 255.0
    y = np.array(y)

    return X, y


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

