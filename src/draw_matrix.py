import matplotlib.pyplot as plt
import dill as pickle
import sys
import numpy as np
sys.path.append('../thirdParty')


def draw_matrix(matrix, title=None, save_path=None):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='viridis')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


if __name__ == "__main__":
    with open('ZZ_dataDirs/topoOptim_05112024151343/optimOut.plat', 'rb') as file:
        [population, result, name, problem] = pickle.load(file, encoding="latin1")

    # from result take the best solution (minimal x.objectives[0] and x.objectives[1]) and show its variables
    sorted_result = sorted(result, key=lambda x: x.objectives[0] + x.objectives[1])
    for i in range(5):
        draw_matrix(np.array(sorted_result[i].variables[0]).reshape((10, 52)), title=f'solution {i}')