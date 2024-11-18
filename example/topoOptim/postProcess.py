import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def plot_pareto_fronts(archive, n_gens, pop_size, output_file='pareto_fronts.png', limit_x=10, limit_y=10):
    plt.figure(figsize=(16, 9))
    n_gens += 1
    # Separates the archive into generations
    archive = [archive[i:i + pop_size] for i in range(0, n_gens * pop_size, pop_size)]
    colors = cm.rainbow(np.linspace(0, 1, len(archive)))
    for i in range(n_gens):
        front = archive[i]
        objectives = [[sol.objectives[0], sol.objectives[1]] for sol in front]
        objectives = [sol for sol in objectives if (limit_x is None or sol[0] < limit_x) and (limit_y is None or sol[1] < limit_y)]
        plt.scatter([sol[0] for sol in objectives], [sol[1] for sol in objectives], color=colors[i])

    plt.xlabel("pressureRecoveryFactor")
    plt.ylabel("uniformityIndex")
    plt.legend([f"G{i}" for i in range(n_gens)], bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title("Pareto fronts over generations")
    plt.savefig(output_file)
    plt.show()


if __name__ == '__main__':
    import dill as pickle
    import sys
    sys.path.append('../../thirdParty')

    # Load the data
    # with open('ZZ_dataDirs/topoOptim_14112024145812/optimOut.plat', 'rb') as file:
    #     [population, result, name, problem] = pickle.load(file, encoding="latin1")
    data_dir = 'ZZ_dataDirs/topoOptim_18112024122615/'
    with open(f'{data_dir}archive.plat', 'rb') as file:
        [archive, n_gens, pop_size, n_iter] = pickle.load(file, encoding="latin1")

    plot_pareto_fronts(archive, n_gens, pop_size, output_file=f'{data_dir}pareto_fronts-limit10.png', limit_x=10, limit_y=10)
    plot_pareto_fronts(archive, n_gens, pop_size, output_file=f'{data_dir}pareto_fronts-limit0.png', limit_x=0, limit_y=0)
    plot_pareto_fronts(archive, n_gens, pop_size, output_file=f'{data_dir}pareto_fronts-nolimit.png', limit_x=None, limit_y=None)

    # Extract the optimal solutions
    # plot_data(None, result, figname="allData.png", limit=150)


    # TODO: vyvoj pareto fronty pres generace (platypus Archive)
    # vykreslit x nejlepsich z referencnich
