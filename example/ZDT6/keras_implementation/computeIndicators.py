import os
import sys
import csv
import numpy as np
import dill as pickle
from matplotlib import cm
import matplotlib.pyplot as plt
from CFDNNetAdaptV4 import CFDNNetAdapt

sys.path.insert(1, "../../src")
from testFunctions import *
sys.path.insert(1, "../../thirdParty")
import platypusModV2 as plat

# parameters
runDir = "01_algoRuns/run_16/"
xName = "f1"
yName = "f2"
logName = "log.out"
parName = "optimOut.plat"
igdName = "31_igdValues.dat"
hvName = "32_hvValues.dat"

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

# prepare hypervolume indicator
indicator = plat.Hypervolume(minimum=smpMins, maximum=smpMaxs)

# read nsgaii data -- inverted generational distance
fileName = algorithm.smpDir + algorithm.prbDir + igdName
with open(fileName, 'r') as file:
    reader = csv.reader(file)

    cols = next(reader)

    data = list()
    for line in reader:
        data.append([float(i) for i in line])

xI = cols.index("step")
yI = cols.index("igd")

data = np.array(data)
ax1.scatter(data[:, xI], data[:, yI], label="NSGA-II", color="black", marker="x")

# read nsgaii data -- hypervolume
fileName = algorithm.smpDir + algorithm.prbDir + hvName
with open(fileName, 'r') as file:
    reader = csv.reader(file)

    cols = next(reader)

    data = list()
    for line in reader:
        data.append([float(i) for i in line])

xI = cols.index("step")
yI = cols.index("diffToOptimal")

data = np.array(data)
ax2.scatter(data[:, xI], data[:, yI], label="NSGA-II", color="black", marker="x")

# prepare and plot optimal solution
optSolsIGD = optSolsZDT6(10, algorithm.nPars)
optSolsHV = optSolsZDT6(3000, algorithm.nPars)
for i in range(len(optSolsHV)):
    f1, f2 = ZDT6(optSolsHV[i])
    optSolsHV[i].append(f1)
    optSolsHV[i].append(f2)

for i in range(len(optSolsHV)):
    optSolsHV[i][algorithm.nPars + 0] = (optSolsHV[i][algorithm.nPars + 0] - smpMins[algorithm.nPars + 0]) / (
                smpMaxs[algorithm.nPars + 0] - smpMins[algorithm.nPars + 0])
    optSolsHV[i][algorithm.nPars + 1] = (optSolsHV[i][algorithm.nPars + 1] - smpMins[algorithm.nPars + 1]) / (
                smpMaxs[algorithm.nPars + 1] - smpMins[algorithm.nPars + 1])

# prepare problem -- hypervolume
problem = plat.Problem(algorithm.nPars, algorithm.nObjs)
problem.types[:] = [plat.Real(0.0, 1.0)] * algorithm.nPars

# compute hypervolume for optimal solutions
popData = list()
for solution in optSolsHV:
    individuum = plat.core.Solution(problem)
    individuum.variables = [solution[i] for i in range(algorithm.nPars)]
    individuum.objectives = [solution[i] for i in range(algorithm.nPars, len(solution))]
    individuum.evaluated = True
    popData.append(individuum)

optHV = indicator.calculate(popData)

# read cfdnnetadapt log
fileName = runDir + logName
with open(fileName, 'r') as file:
    data = file.readlines()

# get the best DNNs from each iteration
sizes = list()
bestDNNs = list()
for line in data:
    if "Best DNN found " in line:
        bestDNNs.append(line.split()[-1])

    elif "Using" in line:
        sams = line.split()
        trainSamples = int(sams[2])
        valSamples = int(sams[5])
        testSamples = int(sams[9])
        num = trainSamples + valSamples + testSamples
        sizes.append(num)

# go over steps and plot
igds = list()
hvs = list()
for n in range(len(bestDNNs)):
    stepDir = "step_%04d/" % (n + 1)
    fileName = runDir + stepDir + bestDNNs[n] + "/" + parName

    # read data from optimization
    with open(fileName, 'rb') as file:
        [population, result, name, problem] = pickle.load(file, encoding="latin1")

    # process data
    data = list()
    for i in range(len(result)):
        netPars = result[i].variables[:]
        netOuts = result[i].objectives[:]

        # concatenate
        point = netPars + netOuts

        # true values
        recx, recy = ZDT6(point[:algorithm.nPars])

        # descale
        recx = (recx - smpMins[algorithm.nPars + 0]) / (smpMaxs[algorithm.nPars + 0] - smpMins[algorithm.nPars + 0])
        recy = (recy - smpMins[algorithm.nPars + 1]) / (smpMaxs[algorithm.nPars + 1] - smpMins[algorithm.nPars + 1])

        # add
        point.append(recx)
        point.append(recy)
        data.append(point)

    data = np.array(data)

    # compute inverted generational distance
    igd = 0.0
    dists = list()
    for i in range(len(optSolsIGD)):
        dist = 100.0
        for j in range(len(data)):
            nondom = np.array(optSolsIGD[i][:algorithm.nPars])
            netple = np.array(data[j][:algorithm.nPars])

            aux = np.linalg.norm(nondom - netple)
            if aux < dist:
                dist = aux

        dists.append(dist)
        igd += dist

    igd /= len(optSolsIGD)
    igds.append(igd)

    # hypervolume indicator for true values
    popData = list()
    for solution in data:
        individuum = plat.core.Solution(problem)
        individuum.variables = [solution[i] for i in range(algorithm.nPars)]
        individuum.objectives = [solution[i] for i in range(algorithm.nPars + algorithm.nObjs, len(solution))]
        individuum.evaluated = True
        popData.append(individuum)

    hv = indicator.calculate(popData)
    hvs.append(hv)

# differences to optimal hypervolume
diffs = list()
for i in range(len(hvs)):
    diff = abs(optHV - hvs[i])
    diffs.append(diff)

# plot
ax1.scatter(sizes, igds, label="CFDNNetAdapt")
ax1.legend()

ax1.set_yscale("symlog", base=10, linthresh=1e-3)
ax1.set_xlabel("number of samples")
ax1.set_ylabel("mean inverted generational distance (IGD)")

ax2.scatter(sizes, diffs, label="CFDNNetAdapt")
ax2.legend()

ax2.set_yscale("symlog", base=10, linthresh=1e-3)
ax2.set_xlabel("number of samples")
ax2.set_ylabel("difference to optimal hypervolume")

plt.savefig(runDir + "indicators.png")
plt.close()
