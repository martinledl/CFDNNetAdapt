import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_data(all_solutions, optimal, figname="allData.png", limit=None):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    if all_solutions is not None:
        # Sort all solutions based on the sum of x and y
        all_solutions_sorted = sorted(all_solutions, key=lambda sol: sol[0] + sol[1])
        xs_all_sorted = [sol[0] for sol in all_solutions_sorted]
        ys_all_sorted = [sol[1] for sol in all_solutions_sorted]

        # Normalize colors based on the sorted position
        colors_all = np.linspace(0, 1, len(xs_all_sorted))  # Create a range of colors

        # Plot sorted all solutions with color based on their position
        scatter_all = ax.scatter(xs_all_sorted, ys_all_sorted, c=colors_all, cmap='viridis', marker="x", label="all")

    # For the optimal solutions
    xs_optimal = list()
    ys_optimal = list()

    for i in range(len(optimal)):
        netOuts = optimal[i].objectives[:]
        xs_optimal.append(netOuts[0])
        ys_optimal.append(netOuts[1])

    # Sort optimal solutions based on the sum of x and y
    end = len(optimal) if limit is None else min(limit, len(optimal))
    optimal_sorted = sorted(zip(xs_optimal, ys_optimal), key=lambda sol: sol[0] + sol[1])
    xs_optimal_sorted = [sol[0] for sol in optimal_sorted[:end]]
    ys_optimal_sorted = [sol[1] for sol in optimal_sorted[:end]]

    # Normalize colors for optimal solutions
    colors_optimal = np.linspace(0, 1, len(xs_optimal_sorted))  # Create a range of colors

    # Plot sorted optimal solutions with color based on their position
    scatter_optimal = ax.scatter(xs_optimal_sorted, ys_optimal_sorted, c=colors_optimal, cmap='viridis', marker="o",
                                 label="optimal")

    ax.set_xlabel("pressureRecoveryFactor")
    ax.set_ylabel("uniformityIndex")

    plt.colorbar(scatter_optimal, ax=ax, label='Sorted Position')  # Add color bar for reference
    plt.savefig(figname)
    plt.show()


if __name__ == '__main__':
    import dill as pickle
    import sys
    sys.path.append('../../thirdParty')

    # Load the data
    with open('ZZ_dataDirs/topoOptim_08112024160418/optimOut.plat', 'rb') as file:
        [population, result, name, problem] = pickle.load(file, encoding="latin1")

    # Extract the optimal solutions
    optimal = result
    plot_data(None, optimal, figname="allData.png", limit=150)
