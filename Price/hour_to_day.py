import pandas as pd

# #script to resample hour to day
coins = {}
coin_names = ['BTC-USD', 'ETH-USD', 'LTC-USD','BCH-USD', 'XLM-USD', 'XRP-USD']
coin_dirs = []
for i in range(len(coin_names)): coin_dirs.append('./datasets/coinbase_hour_candles/' + coin_names[i] + '.feather')
coins = {coin_names[i]:coin_dirs[i] for i in range(len(coin_names))}

for coin in coins:
    candle = pd.read_feather(coins[coin])
    candle.set_index('time', inplace=True)
    candle.index = candle.index.tz_localize(None)
    df = candle.loc[:, ['open', 'high', 'low', 'close']]
    df_open = df.open.resample('D').first().ffill()
    df_high = df.high.resample('D').max().ffill()
    df_low = df.low.resample('D').min().ffill()
    df_close = df.close.resample('D').last().ffill()
    df = pd.concat([df_open, df_high, df_low, df_close], axis=1)
    df = df.reset_index()
    df.to_feather('./datasets/coinbase_day_candles/' + coin + '.feather')