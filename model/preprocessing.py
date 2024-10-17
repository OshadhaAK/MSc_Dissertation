# model/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input(data):
    """
    Preprocess input data from an array.
    Assumes 'data' is a list of values.
    """
    scaler = StandardScaler()
    data = np.array(data).reshape(1, -1)
    return scaler.fit_transform(data)

def preprocess_csv(file):
    """
    Preprocess input data from a CSV file.
    Assumes the CSV has the relevant features.
    """
    df = pd.read_csv(file)
    df['DateTime'] = pd.to_datetime(df['Time'])
    df = df.drop(["Time"],axis=1)
    df.index = pd.to_datetime(df["DateTime"])
    df = df.drop(["DateTime"],axis=1)
    df["mon"] = df.index.month
    df["day"] = df.index.weekday
    df[["movave_1", "movstd_1"]] = df["Power consumption"].rolling(1440).agg([np.mean, np.std])
    df[["movave_7", "movstd_7"]] = df["Power consumption"].rolling(1440*7).agg([np.mean, np.std])
    df[["movave_30", "movstd_30"]] = df["Power consumption"].rolling(1440*30).agg([np.mean, np.std])
    mean = np.mean(df['Power consumption'].values)
    std = np.std(df['Power consumption'].values)
    data_rolling = df['Power consumption'].rolling(window=1440*7)
    df['q10'] = data_rolling.quantile(0.1).to_frame("q10")
    df['q50'] = data_rolling.quantile(0.5).to_frame("q50")
    df['q90'] = data_rolling.quantile(0.9).to_frame("q90")
    df["target"] = df['Power consumption'].add(-mean).div(std)
     
    features = []
    targets = []
    tau =  1440*30 #forecasting periods
     
    for t in range(1, tau+1,1440):
       df["target_t" + str(t)] = df.target.shift(-t)
       targets.append("target_t" + str(t))
        
    for t in range(1,tau+1,1440):
       df["feat_ar" + str(t)] = df.target.shift(t)
       features.append("feat_ar" + str(t))
        
    for t in [1440*1, 1440*7, 1440*30]:
       df[["feat_movave" + str(t), "feat_movstd" + str(t), "feat_movmin" + str(t) ,"feat_movmax" + str(t)]] = df["Power consumption"].rolling(t).agg([np.mean, np.std, np.max, np.min])
       features.append("feat_movave" + str(t))
       features.append("feat_movstd" + str(t))
       features.append("feat_movmin" + str(t))
       features.append("feat_movmax" + str(t))
        
    months = pd.get_dummies(df.mon, prefix="mon", drop_first=True)
    months.index = df.index
    df = pd.concat([df, months], axis=1)

    days = pd.get_dummies(df.day, prefix="day", drop_first=True)
    days.index = df.index
    df = pd.concat([df, days], axis=1)

    features = features + months.columns.values.tolist() + days.columns.values.tolist()
     
    data_feateng = df[features + targets].dropna()
     
    return data_feateng,features,targets, mean, std
