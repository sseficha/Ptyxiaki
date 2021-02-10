import argparse

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import pandas as pd

# parser = argparse.ArgumentParser()
# parser.add_argument("--threshold", help="give threshold for labels",
#                     type=float, nargs='+')
# args = parser.parse_args()
# print('Run for ',args.threshold,' threholds')

res = [['acc', 'val_acc'], ['loss', 'val_loss'], ['pnl', 'val_pnl']]


fig = plt.figure()
gs = fig.add_gridspec(3, 2)
fig.suptitle('Results')

for i in range(len(res)):
    for j in range(len(res[0])):
        with open('./pickles/'+res[i][j]+".pkl", "rb") as f:
            dic = pickle.load(f)
            #smooth
            dic = pd.DataFrame(dic)
            dic = dic.rolling(5, center=True).mean()
            ax = fig.add_subplot(gs[i, j])
            sb.lineplot(data=dic, ax=ax, dashes=False)
            ax.set_title(res[i][j])


plt.show()
