import pandas as pd
from plotly import graph_objs as go
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


class Evaluator:

    @staticmethod
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


    @staticmethod #return unrealised pnl
    def get_pnl(y_pred, df_candle):
        pnl = Evaluator.pnl_from_positions(df_candle, y_pred, commission=0.01)
        pnl = pnl.cumsum()
        return pnl

    @staticmethod
    def evaluate(y_pred, y_test, df_candle, force=False):
        if force:
            y_predf = y_pred.copy()
            if y_predf.iloc[0] == 0:
                y_predf.iloc[0] = 1
            for i in range(len(y_predf)):
                if y_predf.iloc[i] == 0:
                    y_predf.iloc[i] = y_predf.iloc[i - 1]
            pnl = Evaluator.get_pnl(y_predf, df_candle)
        else:
            pnl = Evaluator.get_pnl(y_pred, df_candle)
        # # confusion matrix
        # conf_m = confusion_matrix(y_test.to_numpy(), y_pred.to_numpy())
        # conf_m = np.array([(i / np.sum(i)) * 100 for i in conf_m])  # turn to percentage
        # heatmap(data=conf_m, annot=True, cmap='Blues', xticklabels=['sell', 'out', 'buy'],
        #         yticklabels=['sell', 'out', 'buy'], fmt='.2f')

        # Plot the cumulative pnl
        # fig = go.Figure()
        # fig.add_scatter(y=pnl, x=pnl.index)
        # fig.show()
        # if force:
        #     z = pd.concat([df_candle.close, y_predf], axis=1)
        # else:
        #     z = pd.concat([df_candle.close, y_pred], axis=1)
        # z.rename(columns={0: 'action'}, inplace=True)
        # fig = go.Figure()
        # fig.add_scatter(x=z.index, y=z.close, mode='lines+markers',
        #                 marker={'color': z.action, 'colorscale': 'Bluered'}, text=z.action)
        # fig.show()

        plt.figure(figsize=(10,12))
        ax1 = plt.subplot(3,1,1)
        sb.lineplot(x=pnl.index, y=pnl)
        ax2 = plt.subplot(3,1,2)
        if force:
            z = pd.concat([df_candle.close, y_predf], axis=1)
        else:
            z = pd.concat([df_candle.close, y_pred], axis=1)
        z.rename(columns={0: 'action'}, inplace=True)
        sb.lineplot(data=z, x=z.index, y='close')
        sb.scatterplot(data=z, x=z.index, y='close', hue='action', s=30, palette=['red','blue','green'])
        ax3 = plt.subplot(3,2,5)
        # confusion matrix
        conf_m = confusion_matrix(y_test.to_numpy(), y_pred.to_numpy())
        conf_m = np.array([(i / np.sum(i)) * 100 for i in conf_m])  # turn to percentage
        sb.heatmap(data=conf_m, annot=True, cmap='Blues', xticklabels=['sell', 'out', 'buy'],
                yticklabels=['sell', 'out', 'buy'], fmt='.2f')
        plt.tight_layout()
        plt.show()

