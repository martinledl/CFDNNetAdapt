# import
import os
import sys
import csv
import numpy as np
import dill as pickle
from matplotlib import cm
import matplotlib.pyplot as plt
from CFDNNetAdaptV4 import CFDNNetAdapt
from src.testFunctions import optSolsZDT6, ZDT6

# parameters
runDir = "01_algoRuns/run_13/"
xName = "f1"
yName = "f2"
logName = "log.out"
parName = "optimOut.plat"

# prepare CFDNNetAdapt
algorithm = CFDNNetAdapt()

# problem specification
algorithm.nPars = 2
algorithm.nObjs = 2
algorithm.nOuts = 2
algorithm.mainDir = "01_algoRuns/"
algorithm.smpDir = "00_prepData/"
algorithm.prbDir = "ZDT6/"
algorithm.dataNm = "10_platypusAllSolutions.dat"
algorithm.minMax = ""

# prepare plot
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# read scales
smpMins, smpMaxs = algorithm.getScalesFromFile(algorithm.smpDir + algorithm.prbDir, algorithm.dataNm)

# prepare and plot optimal solution
optSols = optSolsZDT6(100, algorithm.nPars)
f1s = list()
f2s = list()
for i in range(len(optSols)):
    f1, f2 = ZDT6(optSols[i])
    f1s.append(f1)
    f2s.append(f2)
ax1.plot(f1s, f2s, label="optimal solution", color="black")
ax2.plot(f1s, f2s, label="optimal solution", color="black")

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
ax1.scatter(xs, ys, label="NSGA-II", color="black", marker="x")
ax2.scatter(xs, ys, label="NSGA-II", color="black", marker="x")

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
    recxs = list()
    recys = list()
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

        # true values
        recx, recy = ZDT6(data[:algorithm.nPars])
        recxs.append(recx)
        recys.append(recy)

    ax1.scatter(xs, ys, label=bestDNNs[n], color=colors[n])
    ax2.scatter(recxs, recys, label=bestDNNs[n], color=colors[n])

# finish plot
ax1.set_xlabel(xName)
ax2.set_ylabel(yName)

ax1.set_title("predicted space")
ax2.set_title("recomputed space")

plt.legend()
plt.savefig("objSpacePlot.png")
plt.show()
plt.close()
