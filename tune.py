import sys
import pandas as pd
from sklearn.metrics import classification_report
from pckgs.models import get_model_mlp, get_model_single_lstm, \
    train_model, get_model_double_lstm, get_model_both_emb, get_model_cnn
from pckgs.price_preprocess import PricePreprocess
import math
import numpy as np
import argparse
from pckgs.helper import reduce
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", help="give threshold for labels",
                    type=float, nargs='+')
parser.add_argument("--problem", help="type of input",
                    type=str)
args = parser.parse_args()
print('Run for ',args.threshold,' threholds')
print('Run for ',args.problem,' problem')


candle = pd.read_feather('./Price/datasets/coinbase_day_candles/BTC-USD.feather')
# candle = pd.read_feather('./Price/datasets/coinbase_hour_candles/BTC-USD.feather')
candle.set_index('time', inplace=True)

###
sentiment = pd.read_csv('./Text/datasets/headline_sentiment_mean.csv', index_col='date', parse_dates=['date'])
###
sentiment_score = reduce(sentiment, lag=21)
sentiment_score.dropna(inplace=True)
sentiment_score.drop('sentiment_score_t', axis=1, inplace=True)


start_timestamp = '2015-01-01 00:00:00'
split_timestamp = '2019-01-30 00:00:00'
end_timestamp = '2019-12-31 23:00:00'
lag = 21

# acc = {}
# val_acc = {}
# loss = {}
# val_loss = {}
# pnl = {}
# val_pnl = {}

res = {'acc':{}, 'val_acc':{}, 'loss':{}, 'val_loss':{}, 'pnl':{}, 'val_pnl':{}}

for threshold in args.threshold:
    pp = PricePreprocess(lag, threshold)

    positions = pp.preprocess(candle)
    y = positions.loc[:, ['down', 'same', 'up']]
    if args.problem == 'p':
        x = positions.drop(['down', 'same', 'up'], axis=1)
    elif args.problem == 's':
        x = sentiment_score
    elif args.problem == 'ps':
        x = positions.drop(['down', 'same', 'up'], axis=1)
        x = sentiment_score.merge(x, left_index=True, right_index=True, how='right').dropna()
    x_train = x.loc[(start_timestamp <= x.index) & (x.index <= split_timestamp)]
    y_train = y.loc[(start_timestamp <= y.index) & (y.index <= split_timestamp)]
    x_test = x.loc[(split_timestamp < x.index) & (x.index <= end_timestamp)]
    y_test = y.loc[(split_timestamp < y.index) & (y.index <= end_timestamp)]

    if args.problem == 'ps':
        x_train = x_train.values.reshape((len(x_train), int(len(x_train.columns) / 2), 2), order='F')
        x_test = x_test.values.reshape((len(x_test), int(len(x_test.columns) / 2), 2), order='F')

    test_index = y_test.index
    test_candle = candle.reindex(test_index)

    train_index = y_train.index
    train_candle = candle.reindex(train_index)

    if args.problem == 'p' or args.problem == 's':
        model = get_model_single_lstm()
    elif args.problem == 'ps':
        model = get_model_double_lstm()
    model, model_pnl, history, pnl_test, pnl_train = train_model(model, (x_train, x_test, y_train, y_test),train_candle, test_candle,
                                                                 './models/model_price.h5', epochs=10)

    res['acc'][threshold] = history.history['accuracy']
    res['val_acc'][threshold] = history.history['val_accuracy']
    res['loss'][threshold] = history.history['loss']
    res['val_loss'][threshold] = history.history['val_loss']
    res['pnl'][threshold] = pnl_train
    res['val_pnl'][threshold] = pnl_test


    # ac = {'accuracy': history.history['accuracy'], 'val_accuracy': history.history['val_accuracy']}
    # loss = {'loss': history.history['loss'], 'val_loss': history.history['val_loss']}
    # res = {'accuracy': ac, 'loss': loss, 'pnl': pnl}

    # a_file = open('./pickles/'+str(threshold)+".pkl", "wb")
    # pickle.dump(res, a_file)
    # a_file.close()

# file = open('./pickles/acc.pkl', "wb")
# pickle.dump(acc, file)
# file.close
# with open('./pickles/acc.pkl', "wb") as f:
#     pickle.dump(acc, f)
for key in res:
    with open('./pickles/'+key+'.pkl', "wb") as f:
        pickle.dump(res[key], f)