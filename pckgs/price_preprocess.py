import pandas as pd
import datetime
import seaborn as sb
import matplotlib.pyplot as plt
from pckgs.helper import timeseries_to_supervised, timeseries_to_supervised2
from sklearn.preprocessing import MinMaxScaler
from math import exp

class BitcoinPreprocess:
    def __init__(self, resample, lag, future):
        self.resample = resample
        self.lag = lag
        self.threshold = 0
        self.future = future

    def classify(self, change):
        if change < -self.threshold:
            return 'down'
        elif change > self.threshold:
            return 'up'
        else:
            return 'same'

    def preprocess(self, df):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.index = df.Timestamp
        df = df.loc[(df['Timestamp'] >= datetime.datetime(2016, 1, 1, 00, 00, 00)) &
                    (df['Timestamp'] <= datetime.datetime(2019, 12, 31, 00, 00, 00))]

        df = df.loc[:, ['Close']]  # we will work with the closing price
        df.dropna(inplace=True)
        if self.resample == 'D':
            df = df.resample('D').last()
            self.threshold = 1  # !!!!!!!!
        elif self.resample == 'H':
            df = df.resample('H').last()
            self.threshold = 0.1

        # print("Null values:")
        # print(df.loc[df.Close.isna()])
        df.Close.fillna(method='ffill', inplace=True)

        # print('\n Number of observations')
        # print(df.shape)

        plt.figure(figsize=(10, 5))
        sb.lineplot(data=df.Close)

        #weighted future percentage changes
        for i in range(self.future):
            label = 'pChange{}'.format(i)
            df[label] = ((df.Close.shift(-i) / df.Close.shift(1)) - 1) * 100
            # df[label] = df[label] * exp(-i)


        df.dropna(inplace=True)
        df.drop([label for label in df.columns.values if not label.startswith('pChange')], axis=1, inplace=True)

        #weighted average
        # df['pChange'] = df.sum(axis=1) / sum(exp(-i) for i in range(self.future))
        df['pChange'] = df.sum(axis=1) / self.future

        print(df)
        df = pd.DataFrame(df.loc[:, ['pChange', 'pChange0']])  #pChange0 for x pChange for y

        # a = pd.DataFrame(df.loc[:, 'pChange0'])
        # a.to_csv('../pChange.csv')

        plt.figure(figsize=(10, 5))
        sb.lineplot(data=df['pChange'])

        #scale
        scaler = MinMaxScaler(feature_range=(-1,1))
        df['pChange_scaled'] = scaler.fit_transform(df['pChange0'].values.reshape(-1, 1))


        # create shifted observations
        # df_lagged = timeseries_to_supervised(df, 'pChange_scaled', lag=self.lag)
        df_lagged = timeseries_to_supervised2(pd.DataFrame(df['pChange_scaled']), lag=self.lag)
        df_lagged.drop(columns=['pChange_scaled_t'], inplace=True)


        df = pd.concat([df, df_lagged], axis=1)
        df.drop(columns=['pChange_scaled'], inplace=True)
        df.dropna(inplace=True)

        # make classification problem
        df['pChange'] = df['pChange'].apply(self.classify)
        print('\n Value of observations: \n')
        print(df['pChange'].value_counts())

        # one hot encode
        one_hot_df = pd.get_dummies(df['pChange'])
        df = pd.concat([df, one_hot_df], axis=1)
        df.drop(['pChange', 'pChange0'], axis=1, inplace=True)

        return df


class CandlePreprocess:
    def __init__(self, resample):
        self.resample = resample

    def preprocess(self, df):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.index = df.Timestamp
        df = df.loc[(df['Timestamp'] >= datetime.datetime(2016, 1, 1, 00, 00, 00)) &
                    (df['Timestamp'] <= datetime.datetime(2019, 12, 31, 00, 00, 00))]
        # OHLC
        df = df.loc[:, ['Open', 'High', 'Low', 'Close']]
        df_open = df.Open.resample(self.resample).first()
        df_high = df.High.resample(self.resample).max()
        df_low = df.Low.resample(self.resample).min()
        df_close = df.Close.resample(self.resample).last()

        df_open.fillna(method='ffill', inplace=True)
        df_high.fillna(method='ffill', inplace=True)
        df_low.fillna(method='ffill', inplace=True)
        df_close.fillna(method='ffill', inplace=True)

        df = pd.concat([df_open, df_high, df_low, df_close], axis=1)
        return df

