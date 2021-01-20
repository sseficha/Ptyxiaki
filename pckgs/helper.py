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




# get dataframe df and return a new dataframe with only the specified timeseries label shifted for specified lags
# def timeseries_to_supervised(df, label, lag=1, goal=None):
#     df_sup = df.copy()
#     df_sup = df_sup.drop([i for i in df_sup.columns if i != label], axis=1)
#     df = pd.DataFrame(df[label])
#     for i in range(1, lag, 1):
#     # for i in range(lag-1, 0, -1):
#         temp = df.shift(i)
#         string = label + '_t-' + str(i)  # changed from t to _t
#         temp = temp.rename(columns={label: string})
#         df_sup = pd.concat([df_sup, temp], axis=1)
#     df_sup = df_sup.rename(columns={label: label + '_t'})  # changed from t to _t
#     # goal
#     if goal is not None:
#         temp = df.shift(-goal)
#         temp = temp.rename(columns={label: 'y'})
#         df_sup = pd.concat([df_sup, temp], axis=1)
#     return df_sup

def timeseries_to_supervised2(df, lag):
    shifted_df = df.copy()
    shifted_df.columns = [str(col) + '_t' for col in shifted_df.columns]
    for i in range(1, lag):
        shifted = df.shift(i)
        shifted.columns = [str(col) + '_t-' + str(i) for col in shifted.columns]
        shifted_df = pd.concat([shifted_df, shifted],axis=1)
    return shifted_df


def get_positions(pp, coins):
    positions = pd.DataFrame()
    for coin in coins:
        candle = pd.read_feather(coin)
        candle.set_index('time', inplace=True)
        candle.index = candle.index.tz_localize(None)
        positions = pd.concat([positions, pp.preprocess(candle)])
    return positions

def custom_split(pp, coins, problem, start_timestamp, split_timestamp, end_timestamp):
    positions = get_positions(pp, coins)
    y = positions.loc[:, ['down', 'same', 'up']]
    x = positions.drop(['down', 'same', 'up'], axis=1)

    if problem == 'pp': #turn everything after this into a func
        sentiment = pd.read_csv('../Text/datasets/headline_sentiment_mean.csv', index_col='date', parse_dates=['date'])
        sentiment_score = timeseries_to_supervised2(sentiment, lag=21)
        sentiment_score.dropna(inplace=True)
        sentiment_score.drop('sentiment_score_t', axis=1, inplace=True)
        x2 = sentiment_score
        x = x2.merge(x, left_index=True, right_on=pd.to_datetime(x.index.strftime('%Y-%m-%d')), how='right').dropna()
        x.drop(columns='key_0', inplace=True)
        # x = x.values.reshape((len(x), int(len(x.columns) / 2), 2), order='F')
    # elif problem =='pe':
    # x = positions.drop(['down', 'same', 'up'], axis=1)
    # headline = pd.read_csv('../Text/datasets/headline_embeddings_mean.csv', index_col='date', parse_dates=['date'])
    # x2 = HeadlinePreprocess.shape_vectors(headline, lag, y.index)


    #what to do when I reshape with the indexes???
    # if problem == 'p' or problem == 'pp':
    x_train = x.loc[(start_timestamp <= x.index) & (x.index <= split_timestamp)]
    y_train = y.loc[(start_timestamp <= y.index) & (y.index <= split_timestamp)]
    x_test = x.loc[(split_timestamp < x.index) & (x.index <= end_timestamp)]
    y_test = y.loc[(split_timestamp < y.index) & (y.index <= end_timestamp)]
    if problem == 'pp':
        x_train = x_train.values.reshape((len(x_train), int(len(x_train.columns) / 2), 2), order='F')
        x_test = x_test.values.reshape((len(x_test), int(len(x_test.columns) / 2), 2), order='F')


    # elif problem == 'pe':
    #     x1_train, x1_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    #     x2_train, x2_test, _, _ = train_test_split(x2, y, test_size=0.2, shuffle=False)
    #     del _
    #     x_train = [x1_train, x2_train]
    #     x_test = [x1_test, x2_test]

    print(y_train.shape)
    print(y_test.shape)
    return x_train, y_train, x_test, y_test
