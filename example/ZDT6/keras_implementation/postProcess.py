# import
import os
import sys
import csv
import numpy as np
import dill as pickle
from matplotlib import cm
import matplotlib.pyplot as plt

sys.path.insert(1, "../../../src")
sys.path.insert(1, "../../../thirdParty")
from testFunctions import *
from CFDNNetAdaptV4 import CFDNNetAdapt

# parameters
runDir = "01_algoRuns/run_66/"
xName = "f1"
yName = "f2"
logName = "log.out"
parName = "optimOut.plat"

# prepare CFDNNetAdapt
algorithm = CFDNNetAdapt(nPars=2, nObjs=2, nOuts=2, mainDir=runDir, smpDir="../00_prepData/", prbDir="ZDT6/",
                         doNotCreateRunDir=True)

# prepare plot
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# read scales
smpMins, smpMaxs = algorithm.getScalesFromFile(algorithm.smpDir + algorithm.prbDir, algorithm.dataNm)

# prepare and plot optimal solution
optSols = optSolsZDT6(100, algorithm.nPars)
f1s = list()
f2s = list()
x1s = list()
x2s = list()
for i in range(len(optSols)):
    f1, f2 = ZDT6(optSols[i])
    f1s.append(f1)
    f2s.append(f2)
    x1s.append(optSols[i][0])
    x2s.append(optSols[i][1])
ax1.plot(f1s, f2s, label="optimal solution", color="black")
ax2.plot(f1s, f2s, label="optimal solution", color="black")
ax3.scatter(x1s, x2s, label="optimal solution", color="black", marker="o")
ax4.scatter(x1s, f2s, label="optimal solution", color="black", marker="o")

# read samples
source, target = algorithm.loadAndScaleData(algorithm.smpDir + algorithm.prbDir, algorithm.dataNm, algorithm.nPars,
                                            algorithm.nObjs)

# rescale samples
xs = list()
ys = list()
x1s = list()
x2s = list()
for i in range(len(target[0])):
    xs.append(
        target[0][i] * (smpMaxs[algorithm.nPars + 0] - smpMins[algorithm.nPars + 0]) + smpMins[algorithm.nPars + 0])
    ys.append(
        target[1][i] * (smpMaxs[algorithm.nPars + 1] - smpMins[algorithm.nPars + 1]) + smpMins[algorithm.nPars + 1])
    x1s.append(source[0][i] * (smpMaxs[0] - smpMins[0]) + smpMins[0])
    x2s.append(source[1][i] * (smpMaxs[1] - smpMins[1]) + smpMins[1])

# plot sampels
ax1.scatter(xs, ys, label="NSGA-II", color="black", marker="x")
ax2.scatter(xs, ys, label="NSGA-II", color="black", marker="x")
ax3.scatter(x1s, x2s, label="NSGA-II", color="black", marker="x")
ax4.scatter(x1s, ys, label="NSGA-II", color="black", marker="x")

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
    x1s = list()
    x2s = list()
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

        # parameters
        x1s.append(data[0])
        x2s.append(data[1])

    ax1.scatter(xs, ys, label=bestDNNs[n], color=colors[n])
    ax2.scatter(recxs, recys, label=bestDNNs[n], color=colors[n])
    ax3.scatter(x1s, x2s, label=bestDNNs[n], color=colors[n])
    ax4.scatter(x1s, recys, label=bestDNNs[n], color=colors[n])

# finish plot
ax1.set_xlabel(xName)
ax2.set_xlabel(xName)

ax1.set_ylabel(yName)
ax2.set_ylabel(yName)

ax3.set_xlabel("x0")
ax4.set_xlabel("x0")

ax3.set_ylabel("x1")
ax4.set_ylabel(yName)

ax1.set_title("predicted space")
ax2.set_title("recomputed space")

plt.legend()
plt.savefig(runDir + "objSpacePlot.png")
plt.show()
plt.close()
