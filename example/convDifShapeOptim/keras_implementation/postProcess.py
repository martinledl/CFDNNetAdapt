import sys

sys.path.insert(1, "../../../src")
sys.path.insert(1, "../../../thirdParty")
import numpy as np
import dill as pickle
from matplotlib import cm
from testFunctions import *
from CFDNNetAdaptV4 import CFDNNetAdapt
import matplotlib.pyplot as plt


def postProcess(runDir="01_algoRuns/run_20/", xName="energyEfficiency", yName="totalLength", logName="log.out",
                parName="optimOut.plat"):
    # problem specification
    algorithm = CFDNNetAdapt(nPars=14, nObjs=2, nOuts=1, mainDir="01_algoRuns/", smpDir="../00_prepCFDData/",
                             prbDir="14_LConv2CPLDiff4CP/", dataNm="10_platypusCFDAllSolutions.dat", minMax="",
                             doNotCreateRunDir=True)

    # prepare plot
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    # read scales
    smpMins, smpMaxs = algorithm.getScalesFromFile(algorithm.smpDir + algorithm.prbDir, algorithm.dataNm)

    # read samples
    source, target = algorithm.loadAndScaleData(algorithm.smpDir + algorithm.prbDir, algorithm.dataNm, algorithm.nPars,
                                                algorithm.nObjs)

    # rescale samples
    xs = list()
    ys = list()
    for i in range(len(target[0])):
        xs.append(
            target[0][i] * (smpMaxs[algorithm.nPars + 0] - smpMins[algorithm.nPars + 0]) + smpMins[algorithm.nPars + 0])
        ys.append(
            target[1][i] * (smpMaxs[algorithm.nPars + 1] - smpMins[algorithm.nPars + 1]) + smpMins[algorithm.nPars + 1])

    # plot sampels
    ax.scatter(xs, ys, label="NSGA-II", color="black", marker="x")

    # read cfdnnetadapt log
    fileName = runDir + logName
    with open(fileName, 'r') as file:
        data = file.readlines()

    # get the best DNNs from each iteration
    bestDNNs = list()
    for line in data:
        if "Best DNN found " in line:
            bestDNNs.append(line.split()[-1])

    # prepare colors
    colors = cm.rainbow(np.linspace(0.0, 1.0, len(bestDNNs)))

    # go over steps and plot
    for n in range(len(bestDNNs)):
        stepDir = "step_%04d/" % (n + 1)
        fileName = runDir + stepDir + bestDNNs[n] + "/" + parName

        # read data from optimization
        with open(fileName, 'rb') as file:
            [population, result, name, problem] = pickle.load(file, encoding="latin1")

        # process data
        xs = list()
        ys = list()
        for i in range(len(result)):
            netPars = result[i].variables[:]
            netOuts = result[i].objectives[:]

            # concatenate and descale
            data = netPars + netOuts
            data = np.array(data)
            data = data * (smpMaxs - smpMins) + smpMins

            # values predicted by DNN
            xs.append(data[algorithm.nPars + 0])
            ys.append(data[algorithm.nPars + 1])

        ax.scatter(xs, ys, label=bestDNNs[n], color=colors[n])

    # finish plot
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)

    ax.set_title("predicted space")

    plt.legend()
    plt.savefig(runDir + "objSpacePlot.png")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    postProcess(runDir="01_algoRuns/run_28/")