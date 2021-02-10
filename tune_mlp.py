import sys
import pandas as pd
import os
from sklearn.metrics import classification_report
from pckgs.models import get_model_mlp, get_model_lstm, \
    train_model, get_model_both_emb, get_model_cnn
from pckgs.price_preprocess import PricePreprocess
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sb
from pckgs.helper import reduce
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--coin", help="which coin",
                    type=str)
args = parser.parse_args()


args = parser.parse_args()

if args.coin == 'btc':
    candle = pd.read_feather('./Price/datasets/coinbase_day_candles/BTC-USD.feather')
elif args.coin == 'eth':
    candle = pd.read_feather('./Price/datasets/coinbase_day_candles/ETH-USD.feather')

candle.set_index('time', inplace=True)
sentiment = pd.read_csv('./Text/datasets/headline_sentiment_mean.csv', index_col='date', parse_dates=['date'])
sentiment_score = reduce(sentiment, lag=21)
sentiment_score.dropna(inplace=True)
sentiment_score.drop('sentiment_score_t', axis=1, inplace=True)

start_timestamp = '2015-01-01 00:00:00'
split_timestamp = '2018-12-31 00:00:00'
end_timestamp = '2019-12-31 00:00:00'
lag = 21

problems = ['p', 's', 'ps']
thresholds = [0.1, 0.25, 0.5, 0.7]

for problem in problems:
    for threshold in thresholds:

        pp = PricePreprocess(lag, threshold)

        positions = pp.preprocess(candle)
        y = positions.loc[:, ['down', 'same', 'up']]
        if problem == 'p':
            x = positions.drop(['down', 'same', 'up'], axis=1)
        elif problem == 's':
            x = sentiment_score
        elif problem == 'ps':
            x = positions.drop(['down', 'same', 'up'], axis=1)
            x = sentiment_score.merge(x, left_index=True, right_index=True, how='right').dropna()
        x_train = x.loc[(start_timestamp <= x.index) & (x.index <= split_timestamp)]
        y_train = y.loc[(start_timestamp <= y.index) & (y.index <= split_timestamp)]
        x_test = x.loc[(split_timestamp < x.index) & (x.index <= end_timestamp)]
        y_test = y.loc[(split_timestamp < y.index) & (y.index <= end_timestamp)]


        test_index = y_test.index
        test_candle = candle.reindex(test_index)

        train_index = y_train.index
        train_candle = candle.reindex(train_index)

        lrs = [5e-2, 1e-3, 5e-3]
        neurons = [8, 16, 32, 64]
        layers = [1,2,3]
        dropouts = [0.1, 0.2, 0.4]
        for neuron in neurons:
            for lr in lrs:
                for layer in layers:
                    for dropout in dropouts:
                        res = {}
                        model = get_model_mlp(layer, neuron, lr, dropout)
                        history, pnl_test, pnl_train, pnls_test, pnls_train = train_model(model, (x_train, x_test, y_train, y_test),
                                                                                     train_candle, test_candle,
                                                                                      epochs=500)
                        # res['acc'] = history.history['accuracy']
                        # res['val_acc'] = history.history['val_accuracy']
                        res['acc'] = {'acc': history.history['accuracy'], 'val_acc': history.history['val_accuracy']}
                        # res['loss'] = history.history['loss']
                        # res['val_loss'] = history.history['val_loss']
                        res['loss'] = {'loss':history.history['loss'], 'val_loss': history.history['val_loss']}
                        res['pnl'] = pnl_train
                        res['val_pnl'] = pnl_test


                        fig = plt.figure(figsize=(15,15))
                        gs = fig.add_gridspec(3, 2)
                        fig.suptitle('Results for '+str(threshold)+' threshold')

                        for item,i in zip(res.items(), range(len(res))):
                            ax = fig.add_subplot(gs[math.floor(i/2), i%2])
                            sb.lineplot(data=item[1], ax=ax, dashes=False)
                            ax.set_title(item[0])
                        ax = fig.add_subplot(gs[2, 0])
                        ax.set_title('time pnl')
                        plt.xticks(rotation=65)
                        sb.lineplot(data=pnls_train, ax=ax, dashes=False)
                        ax = fig.add_subplot(gs[2, 1])
                        ax.set_title('time val_pnl')
                        plt.xticks(rotation=65)
                        sb.lineplot(data=pnls_test, ax=ax, dashes=False)
                        path = './pickles/{}/mlp/{}/{}/'.format(args.coin, problem, threshold)
                        try:
                            os.makedirs(path)
                        except OSError:
                            pass
                        fig.savefig(path+'{}_{}_{}_{}'.format(layer,neuron,lr, dropout)+'.png')
