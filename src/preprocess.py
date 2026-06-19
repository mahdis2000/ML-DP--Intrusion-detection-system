import pandas as pd
from sklearn.preprocessing import StandardScaler
def preprocess_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)
