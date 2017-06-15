import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def lab1():
    bh = pd.read_csv('data/bh_seq.csv', header=None)
    nn = pd.read_csv('data/nn_seq.csv', header=None)

    bh['totalT'] = bh[[3,4,5]].sum(axis=1)  # building, simulation, deleting
    bh['bs'] = bh[[3,4]].sum(axis=1)        # building, simulation

    bh_T_median = bh.groupby(0, as_index=False)[[4, 'bs', 'totalT']].median()
    nn_T_median = nn.groupby(0, as_index=False)[3].median()

    nn_100k = nn[nn[0] == 10**5][nn[1] == 0]
    bh_100k = bh[bh[0] == 10**5][bh[1] == 0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = nn_T_median.plot.line(x=0, y=3, label='native')
    ax = bh_T_median.plot.line(x=0, y='totalT', label='[bh] building, simulation, deleting', ax=ax)
    ax = bh_T_median.plot.line(x=0, y='bs', label='[bh] building, simulation', ax=ax)
    ax = bh_T_median.plot.line(x=0, y=4, label='[bh] simulation', ax=ax)
    ax.set(xlabel='bodies', ylabel='time, s')
    plt.savefig('1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = bh_100k.plot.line(x=2, y='totalT', xticks=bh_100k[2])
    ax.set(xlabel='simulation step', ylabel='time, s')
    plt.savefig('2.png')


if __name__ == '__main__':
    lab1()
