import pandas as pd
import datetime
import seaborn as sb
import matplotlib.pyplot as plt
from pckgs.helper import timeseries_to_supervised2
from sklearn.preprocessing import MinMaxScaler
from math import exp

class PricePreprocess:
    def __init__(self, lag, threshold, unit=None):
        self.unit = unit
        self.lag = lag
        self.threshold = threshold

    def classify(self, change):
        if change < -self.threshold:
            return 'down'
        elif change > self.threshold:
            return 'up'
        else:
            return 'same'

    def preprocess(self, df):
        df = df.loc[:,['close']]
        #resample
        if self.unit is not None:
            df = df.resample(self.unit).last()
            df.close.fillna(method='ffill', inplace=True)
        #percentage change
        df['pChange'] = ((df.close / df.close.shift(1)) - 1) * 100
        df.drop(columns=['close'], inplace=True)
        #scale
        scaler = MinMaxScaler(feature_range=(-1,1))
        df['pChange_scaled'] = scaler.fit_transform(df['pChange'].values.reshape(-1, 1))
        # create shifted observations
        df_lagged = timeseries_to_supervised2(pd.DataFrame(df['pChange_scaled']), lag=self.lag)
        df_lagged.drop(columns=['pChange_scaled_t'], inplace=True)
        df = pd.concat([df, df_lagged], axis=1)
        df.drop(columns=['pChange_scaled'], inplace=True)
        df.dropna(inplace=True)
        # generate labels
        df['pChange'] = df['pChange'].apply(self.classify)
        print('\n Value of observations: \n')
        print(df['pChange'].value_counts())
        # one hot encode
        df = pd.get_dummies(df, prefix='', prefix_sep='')
        return df


class CandlePreprocess:
    def __init__(self, unit):
        self.unit = unit

    def preprocess(self, df):
        df = df.loc[:, ['Open', 'High', 'Low', 'Close']]
        df_open = df.Open.resample(self.unit).first().ffill()
        df_high = df.High.resample(self.unit).max().ffill()
        df_low = df.Low.resample(self.unit).min().ffill()
        df_close = df.Close.resample(self.unit).last().ffill()
        df = pd.concat([df_open, df_high, df_low, df_close], axis=1)
        return df

