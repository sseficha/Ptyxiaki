import pandas as pd
from sklearn.model_selection import train_test_split
from plotly import graph_objs as go
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
import numpy as np
from scipy import spatial
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec


def pnl_from_positions(candles: pd.DataFrame, positions: pd.Series, commission=0.) -> pd.Series:
    assert candles.shape[0] == positions.shape[0]
    assert type(candles.index) is pd.DatetimeIndex
    assert type(positions.index) is pd.DatetimeIndex
    pos_changes = abs(positions.diff().fillna(positions)) > 0
    pos_prices = candles.open.copy()
    pos_prices[~pos_changes] = pd.NA
    pos_prices = pos_prices.ffill().fillna(candles.open)
    comm_charges = pos_changes * commission
    candle_returns = (candles.close - candles.open) / pos_prices
    step_returns = (candles.open - candles.close.shift().fillna(candles.open)) / pos_prices.shift().bfill()
    total_returns = candle_returns * positions + step_returns * positions.shift().fillna(0)
    pnl = total_returns - comm_charges
    return pnl


# custom callback for doc2vec training
class EpochLogger(CallbackAny2Vec):
    def __init__(self, documents):
        self.epoch = 1
        self.documents = documents

    def on_epoch_end(self, my_model):
        temp_path = get_tmpfile('toy_d2v')
        my_model.save(temp_path)
        my_model = Doc2Vec.load(temp_path)
        if self.epoch >= 5:
            count1 = 0
            count2 = 0
            randoms = np.random.choice(a=np.arange(0, len(self.documents), 1), size=1000, replace=False)
            for i in range(1000):
                doc_id = randoms[i]
                vector = my_model.infer_vector(self.documents[doc_id][0])
                most_sim = my_model.docvecs.most_similar([vector], topn=2)
                most_sim1 = most_sim[0][0]
                if most_sim1 == doc_id: count1 += 1
                most_sim2 = most_sim[1][0]
                if most_sim2 == doc_id: count2 += 1
            print('-----' + str(self.epoch))
            print(str(count1 / 10) + '%')
            print(str(count2 / 10) + '%')
        self.epoch += 1


# custom pnl callback to check on pnl
from tensorflow.keras.callbacks import Callback
from pckgs.evaluator import Evaluator
class PnlCallback(Callback):
    def __init__(self, x_test, df_candle, patience, name):
        self.pnl = -9999
        self.best_epoch = 0
        self.patience = patience
        self.original_patience = patience
        self.x_test = x_test
        self.df_candle = df_candle
        self.name = name
        self.stats = []

    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 10 == 0:
        y_pred = self.model.predict(self.x_test)
        y_pred_labeled = pd.DataFrame(y_pred, columns=[-1, 0, 1], index=self.df_candle.index)
        y_pred_labeled = y_pred_labeled.idxmax(axis=1)
        pnl = Evaluator.get_pnl(y_pred_labeled, self.df_candle)
        pnl = pnl.iloc[len(pnl) - 1]
        self.stats.append(pnl)
        # if pnl greater then set new pnl, save model
        if pnl > self.pnl:
            self.pnl = pnl
            self.best_epoch = epoch
            self.model.save(self.name+'.h5')
            self.patience = self.original_patience
            print('\n Pnl improved, current pnl is: ', self.pnl)
        else:
            self.patience -= 1
        print('\n Pnl has not improved since ',self.best_epoch,' epoch')
        if self.patience <= 0:
            self.model.stop_training = True
            print('\n Reached ', epoch, 'nth epoch! \n')
            print('Pnl is: ', self.pnl)


def min_dist(series):
    if not series.empty:
        length = len(series)
        if length == 1:
            # return series
            return series.values
        else:
            min_id = 0
            min_dist = 99999
            for i in range(length):
                # vector = series.iloc[i]  # list of features ('vec')
                vector = series.iloc[i].values
                dist = 0
                for j in range(length):
                    if j != i:
                        # dist += 1-spatial.distance.cosine(vector, series.iloc[j])
                        dist += 1 - spatial.distance.cosine(vector, series.iloc[j].values)
                if dist < min_dist:
                    min_dist = dist
                    min_id = i
            # return series.iloc[min_id]
            return series.iloc[min_id].values


# get dataframe df and return a new dataframe with only the specified timeseries label shifted for specified lags
def timeseries_to_supervised(df, label, lag=1, goal=None):
    df_sup = df.copy()
    df_sup = df_sup.drop([i for i in df_sup.columns if i != label], axis=1)
    df = pd.DataFrame(df[label])
    for i in range(1, lag, 1):
    # for i in range(lag-1, 0, -1):
        temp = df.shift(i)
        string = label + '_t-' + str(i)  # changed from t to _t
        temp = temp.rename(columns={label: string})
        df_sup = pd.concat([df_sup, temp], axis=1)
    df_sup = df_sup.rename(columns={label: label + '_t'})  # changed from t to _t
    # goal
    if goal is not None:
        temp = df.shift(-goal)
        temp = temp.rename(columns={label: 'y'})
        df_sup = pd.concat([df_sup, temp], axis=1)
    return df_sup

def timeseries_to_supervised2(df, lag):
    shifted_df = df.copy()
    shifted_df.columns = [str(col) + '_t' for col in shifted_df.columns]
    for i in range(1, lag):
        shifted = df.shift(i)
        shifted.columns = [str(col) + '_t-' + str(i) for col in shifted.columns]
        shifted_df = pd.concat([shifted_df, shifted],axis=1)
    return shifted_df

def unscale(y, scaler):
    y_unscaled = y.copy()
    y_unscaled['y'] = scaler.inverse_transform(y_unscaled['y'].values.reshape(-1, 1))
    return y_unscaled


def split(df):
    y = df.loc[:, 'y']
    x = df.drop('y', axis=1)
    # print(x.columns)
    y = pd.DataFrame(y)
    # print(y.columns)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    return x_train, x_test, y_train, y_test
