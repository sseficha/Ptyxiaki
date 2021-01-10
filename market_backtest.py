import pandas as pd
from pathlib import Path
from plotly import graph_objs as go
from fin_utils.labels import oracle_labels
from fin_utils.pnl import pnl_from_positions

if __name__ == '__main__':
    # Load sample data from the folder located in the same directory as this script
    df = pd.read_csv(str(Path(__file__).parent / 'sample_data/eurusd.csv'))
    # Set the index of the dataframe to be the 'date' column
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Create the oracle labels (which if predicted yield the best possible pnl)
    commission = 8e-5
    labels = oracle_labels(df, commission=commission)
	
    print('############################# LABELS ######################')
    print(labels)

    # The commission value of 2e-5 is the standard charge of IBKR. However when trading
    # another cost to keep in mind is that the price you see is not the price you get.
    # When executing a market order to "buy" an asset you are essentially accepting the lowest
    # sell offer available at the exchange, which will be slightly different than the price show.
    # Similarly this happens when you sell. To account for this type of difference, commonly referred
    # to as spread, the commission is slightly increased to encapsulate its cost. Another cost
    # to account for is the slippage, which is the change of price from the moment one places
    # an order up to the moment it is executed, where the price might move in an unwanted direction.
    # This is also accounted for by further increasing the commission.
    # TLDR; Standard commission = 2e-5 + spread + slippage

    pnl = pnl_from_positions(df, labels, commission=commission)
    # The pnl calculated is the pnl of each time step. To get the total pnl, just use sum

    print('############################# PNL ######################')
    print(pnl)

    fig = go.Figure()
    # Plot the cumulative pnl
    fig.add_scatter(y=pnl.cumsum(), x=pnl.index)
    fig.show()
