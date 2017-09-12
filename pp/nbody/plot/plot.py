#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


def lab1():
    bh = pd.read_csv('data/bh_seq.csv', header=None)
    nn = pd.read_csv('data/nn_seq.csv', header=None)

    bh['totalT'] = bh[[3,4,5]].sum(axis=1)  # building, simulation, deleting
    bh['bs'] = bh[[3,4]].sum(axis=1)        # building, simulation

    bh_T_median = bh.groupby(0, as_index=False)[[4, 'bs', 'totalT']].median()
    nn_T_median = nn.groupby(0, as_index=False)[3].median()

    nn_100k = nn[nn[0] == 10**5][nn[1] == 0]
    bh_100k = bh[bh[0] == 10**5][bh[1] == 0]

    nn2bh = (nn_T_median[3]/bh_T_median['totalT']).to_frame()
    nn2bh['bc'] = nn_T_median[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = nn_T_median.plot.line(x=0, y=3, label='native')
    ax = bh_T_median.plot.line(x=0, y='totalT', label='[bh] building, simulation, deleting', ax=ax)
    ax = bh_T_median.plot.line(x=0, y='bs', label='[bh] building, simulation', ax=ax)
    ax = bh_T_median.plot.line(x=0, y=4, label='[bh] simulation', ax=ax)
    ax.set(xlabel='bodies', ylabel='time, s', title='sequential')
    plt.savefig('img/lab1_1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = bh_100k.plot.line(x=2, y='totalT', xticks=bh_100k[2])
    ax.set(xlabel='simulation step', ylabel='time, s', title='sequential')
    plt.savefig('img/lab1_2.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = nn2bh.plot.line(x='bc', legend=False, ax=ax)
    ax.set(xlabel='bodies', ylabel='native/bh', title='sequential')
    plt.savefig('img/lab1_3.png')


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

    nn2bh = (nn_pthread_T_median[3]/bh_pthread_T_median[3]).to_frame()
    nn2bh['bc'] = nn_pthread_T_median[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = nn_T_median.plot.line(x=0, y=3, label='native')
    ax = bh_T_median.plot.line(x=0, y='totalT', label='bh', ax=ax)
    ax = nn_pthread_T_median.plot.line(x=0, y=3, label='native pthread', ax=ax)
    ax = bh_pthread_T_median.plot.line(x=0, y=3, label='bh pthread', ax=ax)
    ax.set(xlabel='bodies', ylabel='time, s', title='pthread')
    plt.savefig('img/lab2_1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = bh_speedup.plot.line(x='bc', y=0, label='bh speedup', ax=ax)
    ax = nn_speedup.plot.line(x='bc', y=3, label='native speedup', ax=ax)
    ax.set(xlabel='bodies', ylabel='speed up', title='pthread')
    plt.savefig('img/lab2_2.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = nn2bh.plot.line(x='bc', legend=False, ax=ax)
    ax.set(xlabel='bodies', ylabel='native/bh', title='pthread')
    plt.savefig('img/lab2_3.png')


def lab3():
    bh = pd.read_csv('data/bh_seq.csv', header=None)
    nn = pd.read_csv('data/nn_seq.csv', header=None)
    bh_omp = pd.read_csv('data/bh_omp.csv', header=None)
    nn_omp = pd.read_csv('data/nn_omp.csv', header=None)

    bh['totalT'] = bh[[3,4,5]].sum(axis=1)  # building, simulation, deleting

    bh_T_median = bh.groupby(0, as_index=False)['totalT'].median()
    nn_T_median = nn.groupby(0, as_index=False)[3].median()
    bh_omp_T_median = bh_omp.groupby(0, as_index=False)[3].median()
    nn_omp_T_median = nn_omp.groupby(0, as_index=False)[3].median()

    bh_speedup = (bh_T_median['totalT']/bh_omp_T_median[3]).to_frame()
    bh_speedup['bc'] = bh_T_median[0]
    nn_speedup = (nn_T_median[3]/nn_omp_T_median[3]).to_frame()
    nn_speedup['bc'] = nn_T_median[0]

    nn2bh = (nn_omp_T_median[3]/bh_omp_T_median[3]).to_frame()
    nn2bh['bc'] = nn_omp_T_median[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = nn_T_median.plot.line(x=0, y=3, label='native')
    ax = bh_T_median.plot.line(x=0, y='totalT', label='bh', ax=ax)
    ax = nn_omp_T_median.plot.line(x=0, y=3, label='native omp', ax=ax)
    ax = bh_omp_T_median.plot.line(x=0, y=3, label='bh omp', ax=ax)
    ax.set(xlabel='bodies', ylabel='time, s', title='omp')
    plt.savefig('img/lab3_1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = bh_speedup.plot.line(x='bc', y=0, label='bh speedup', ax=ax)
    ax = nn_speedup.plot.line(x='bc', y=3, label='native speedup', ax=ax)
    ax.set(xlabel='bodies', ylabel='speed up', title='omp')
    plt.savefig('img/lab3_2.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = nn2bh.plot.line(x='bc', legend=False, ax=ax)
    ax.set(xlabel='bodies', ylabel='native/bh', title='omp')
    plt.savefig('img/lab3_3.png')


def lab4():
    """plotting cuda results"""
    nn = pd.read_csv('data/nn_seq.csv', header=None)
    nn_cuda = pd.read_csv('data/nn_cuda.csv', skiprows=1, header=None)
    nn_omp = pd.read_csv('data/nn_omp.csv', header=None)

    nn_T_median = nn[nn[0] >= 100].groupby(0, as_index=False)[3].median()
    nn_cuda_T_median = nn_cuda.groupby(0, as_index=False)[2].median()
    nn_omp_T_median = nn_omp[nn_omp[0] >= 100].groupby(0, as_index=False)[3].median()

    nn_speedup = (nn_T_median[3]/nn_cuda_T_median[2]).to_frame()
    nn_speedup['bc'] = nn_cuda_T_median[0]
    cuda_vs_omp = (nn_omp_T_median[3]/nn_cuda_T_median[2]).to_frame()
    cuda_vs_omp['bc'] = nn_T_median[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = nn_speedup.plot.line(x='bc', y=0, label='cuda speedup', ax=ax)
    ax.set(xlabel='bodies', ylabel='speed up', title='cuda speedup')
    plt.savefig('img/lab4_1.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = nn_cuda_T_median.plot.line(x=0, y=2, label='native cuda', ax=ax)
    ax.set(xlabel='bodies', ylabel='time, s', title='cuda median time')
    plt.savefig('img/lab4_2.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = cuda_vs_omp.plot.line(x='bc', y=0, ax=ax, legend=False)
    ax.set(xlabel='bodies', title='omp/cuda speedup')
    plt.savefig('img/lab4_3.png')


# plot initial distribution
def _get_body_gen(n, r):
    for i in range(n):
        theta = np.random.uniform(0, np.pi*2)
        u = np.random.uniform(-1, 1)
        a = np.sqrt(1 - u*u)

        body = (
            r * np.cos(theta) * a,
            r * np.sin(theta) * a,
            r * u
        )
        yield body


def _get_sphere(n, r=4000):
    r -= .00001

    points = _get_body_gen(n, r)
    return points


def plot_variance(min_pow=1, max_pow=5):
    npoints = (10**i for i in range(1, max_pow+1))
    for n in npoints:
        sphere = _get_sphere(n)

        fig = plt.figure(n)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*zip(*sphere))

        plt.savefig('sphere_img/{}.png'.format(n))


if __name__ == '__main__':
    # lab1()
    # lab2()
    # lab3()
    lab4()
    # plot_variance()
