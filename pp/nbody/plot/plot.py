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
    plt.savefig('lab1_1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = bh_100k.plot.line(x=2, y='totalT', xticks=bh_100k[2])
    ax.set(xlabel='simulation step', ylabel='time, s')
    plt.savefig('lab1_2.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    nn2bh = (nn_T_median[3]/bh_T_median['totalT']).to_frame()
    nn2bh['bc'] = nn_T_median[0]
    ax = nn2bh.plot.line(x='bc', legend=False, ax=ax)
    ax.set(xlabel='bodies', ylabel='native/bh')
    plt.savefig('lab1_3.png')


def lab2():
    bh = pd.read_csv('data/bh_seq.csv', header=None)
    nn = pd.read_csv('data/nn_seq.csv', header=None)
    bh_pthread = pd.read_csv('data/bh_pthread.csv', header=None)
    nn_pthread = pd.read_csv('data/nn_pthread.csv', header=None)

    bh['totalT'] = bh[[3,4,5]].sum(axis=1)  # building, simulation, deleting

    bh_T_median = bh.groupby(0, as_index=False)['totalT'].median()
    nn_T_median = nn.groupby(0, as_index=False)[3].median()
    bh_pthread_T_median = bh_pthread.groupby(0, as_index=False)[3].median()
    nn_pthread_T_median = nn_pthread.groupby(0, as_index=False)[3].median()

    bh_speedup = (bh_T_median['totalT']/bh_pthread_T_median[3]).to_frame()
    bh_speedup['bc'] = bh_T_median[0]
    nn_speedup = (nn_T_median[3]/nn_pthread_T_median[3]).to_frame()
    nn_speedup['bc'] = nn_T_median[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = nn_T_median.plot.line(x=0, y=3, label='native')
    ax = bh_T_median.plot.line(x=0, y='totalT', label='bh', ax=ax)
    ax = nn_pthread_T_median.plot.line(x=0, y=3, label='native pthread', ax=ax)
    ax = bh_pthread_T_median.plot.line(x=0, y=3, label='bh pthread', ax=ax)
    ax.set(xlabel='bodies', ylabel='time, s')
    plt.savefig('lab2_1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = bh_speedup.plot.line(x='bc', y=0, label='bh speedup', ax=ax)
    ax = nn_speedup.plot.line(x='bc', y=3, label='native speedup', ax=ax)
    ax.set(xlabel='bodies', ylabel='speed up')
    plt.savefig('lab2_2.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    nn2bh = (nn_pthread_T_median[3]/bh_pthread_T_median[3]).to_frame()
    nn2bh['bc'] = nn_pthread_T_median[0]
    ax = nn2bh.plot.line(x='bc', legend=False, ax=ax)
    ax.set(xlabel='bodies', ylabel='native/bh')
    plt.savefig('lab2_3.png')


if __name__ == '__main__':
    lab1()
    lab2()
