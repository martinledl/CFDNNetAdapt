import numpy as np
import pandas as pd
import sys
import dill as pickle
sys.path.append('../../thirdParty')


def hamming_similarity(A, B):
    """Calculates the Hamming similarity between two binary matrices A and B."""
    hamming_distance = np.sum(A != B)
    total_elements = A.size
    return 1 - hamming_distance / total_elements


def jaccard_similarity(A, B):
    """Calculates the Jaccard similarity between two binary matrices A and B."""
    intersection = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return intersection / union


def dice_similarity(A, B):
    """Calculates the Dice similarity between two binary matrices A and B."""
    intersection = np.logical_and(A, B).sum()
    return 2 * intersection / (A.sum() + B.sum())


# Example usage
if __name__ == "__main__":
    with open('ZZ_dataDirs/topoOptim_05112024151343/optimOut.plat', 'rb') as file:
        [population, result, name, problem] = pickle.load(file, encoding="latin1")

    # from result take the best solution (minimal x.objectives[0] and x.objectives[1]) and show its variables
    sorted_result = sorted(result, key=lambda x: x.objectives[0] + x.objectives[1])
    A = np.array(sorted_result[0].variables[0]).reshape((10, 52))

    df = pd.read_csv('00_prepTopoOptimData/feasible52x10.csv')
    sorted_df = df.sort_values(by=["pressureRecoveryFactor", "uniformityIndex"])
    B = np.array(sorted_df.iloc[0, 2:]).reshape((10, 52))

    print("Hamming Similarity:", hamming_similarity(A, B))
    print("Jaccard Similarity:", jaccard_similarity(A, B))
    print("Dice Similarity:", dice_similarity(A, B))
