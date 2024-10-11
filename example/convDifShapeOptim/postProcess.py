
# import
import os
import sys
sys.path.insert(1, "../../src")
sys.path.insert(1, "../../thirdParty")
import csv
import numpy as np
import dill as pickle
from matplotlib import cm
from testFunctions import *
from CFDNNetAdaptV3 import *
import matplotlib.pyplot as plt

# parameters
runDir = "01_algoRuns/run_03/"
xName = "energyEfficiency"
yName = "totalLength"
logName = "log.out"
parName = "optimOut.plat"

# prepare CFDNNetAdapt
algorithm = CFDNNetAdapt()

# problem specification
algorithm.nPars = 14
algorithm.nObjs = 2
algorithm.nOuts = 1
algorithm.mainDir = "01_algoRuns/"
algorithm.smpDir = "00_prepCFDData/"
algorithm.prbDir = "14_LConv2CPLDiff4CP/"
algorithm.dataNm = "10_platypusCFDAllSolutions.dat"
algorithm.minMax = ""

# prepare plot
fig = plt.figure(figsize = (16,9))
ax = fig.add_subplot(111)

# read scales
smpMins, smpMaxs = algorithm.getScalesFromFile(algorithm.smpDir + algorithm.prbDir, algorithm.dataNm)

# read samples
source, target = algorithm.loadAndScaleData(algorithm.smpDir + algorithm.prbDir, algorithm.dataNm, algorithm.nPars, algorithm.nObjs)

# rescale samples
xs = list()
ys = list()
for i in range(len(target[0])):
    xs.append(target[0][i]*(smpMaxs[algorithm.nPars+0] - smpMins[algorithm.nPars+0]) + smpMins[algorithm.nPars+0])
    ys.append(target[1][i]*(smpMaxs[algorithm.nPars+1] - smpMins[algorithm.nPars+1]) + smpMins[algorithm.nPars+1])

# plot sampels
ax.scatter(xs, ys, label = "NSGA-II", color = "black", marker = "x")

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
    stepDir = "step_%04d/" %(n+1)
    fileName = runDir + stepDir + bestDNNs[n] + "/" + parName

    # read data from optimization
    with open(fileName, 'rb') as file:
        [population,result,name,problem] = pickle.load(file, encoding="latin1")

    # process data
    xs = list()
    ys = list()
    for i in range(len(result)):
        netPars = result[i].variables[:]
        netOuts = result[i].objectives[:]

        # concatenate and descale
        data = netPars + netOuts
        data = np.array(data)
        data = data*(smpMaxs - smpMins) + smpMins

        # values predicted by DNN
        xs.append(data[algorithm.nPars+0])
        ys.append(data[algorithm.nPars+1])

    ax.scatter(xs, ys, label = bestDNNs[n], color = colors[n])

# finish plot
ax.set_xlabel(xName)
ax.set_ylabel(yName)

ax.set_title("predicted space")

plt.legend()
plt.savefig(runDir + "objSpacePlot.png")
plt.show()
plt.close()
