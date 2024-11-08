import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_data(all_solutions, optimal, figname="allData.png", limit=None):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    if all_solutions is not None:
        xs = list()
        ys = list()
        for i in range(len(all_solutions)):
            xs.append(all_solutions[i][0])
            ys.append(all_solutions[i][1])

        ax.scatter(xs, ys, color="black", marker="x", label="all")

    xs = list()
    ys = list()
    end = len(optimal) if limit is None else min(limit, len(optimal))
    for i in range(end):
        netOuts = optimal[i].objectives[:]

        # values predicted by DNN
        xs.append(netOuts[0])
        ys.append(netOuts[1])


    if all_solutions is not None:
        ax.scatter(xs, ys, color="red", marker="o", label="optimal")
    else:
        ax.scatter(xs, ys, color="black", marker="x", label="optimal")

    ax.set_xlabel("pressureRecoveryFactor")
    ax.set_ylabel("uniformityIndex")

    plt.savefig(figname)
    plt.show()
