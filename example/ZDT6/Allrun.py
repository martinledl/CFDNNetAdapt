
# import
import os
import sys
sys.path.insert(1, "../../src")
sys.path.insert(1, "../../thirdParty")
import csv
import math
import numpy as np
import dill as pickle
import multiprocessing
from CFDNNetAdaptV3 import *

# bounds
pMin = 0.0
pMax = 1.0

# optimized function
def zdt6(x):
    n = len(x)
    f1 = 1 - math.exp(-4*x[0])*(math.sin(6*math.pi*x[0]))**6
    g = 1 + 9*(sum(x[1:])/(n - 1))**0.25
    f2 = g*(1 - (f1/g)**2)

    return [f1, f2]

# cost function evaluation 
def smpEvaluation(i):
    # evaluate the cases
    checkOut = [0] * (algorithm.nObjs)

    netPars = algorithm.population[i].variables[:]
    netOuts = algorithm.population[i].objectives[:]

    # rescale the pars
    for p in range(len(netPars)):
        netPars[p] = netPars[p]*(algorithm.smpMaxs[p] - algorithm.smpMins[p]) + algorithm.smpMins[p]

    checkOut = zdt6(netPars)

    # rescale for ANN
    for co in range(len(checkOut)):
        checkOut[co] = (checkOut[co] - algorithm.smpMins[algorithm.nPars+co])/(algorithm.smpMaxs[algorithm.nPars+co] - algorithm.smpMins[algorithm.nPars+co])

    # compute the error
    delta = 0
    delta += abs(netOuts[0] - checkOut[0])
    delta += abs(netOuts[1] - checkOut[1])

    algorithm.outFile.write("Doing CFD check no. " + str(algorithm.toCompare.index(i)) + " with parameters " + str(netPars) + "\n")
    algorithm.outFile.write("no. " + str(algorithm.toCompare.index(i)) + " ANN outs were " + str(netOuts) + "\n")
    algorithm.outFile.write("no. " + str(algorithm.toCompare.index(i)) + " CFD outs were " + str(checkOut) + " delta " + str(delta) + "\n")
    algorithm.outFile.write("CFD check no. " + str(algorithm.toCompare.index(i)) + " done\n")
    algorithm.outFile.flush()

    return delta

def dnnEvaluation(vars):
    """ function to return the costs for optimization """
    dummy = 1e6

    # prepare neural network
    netIn = np.array(vars)
    netIn = np.expand_dims(netIn, axis = 1)

    costOut = list()

    for i in range(len(algorithm.nets)):
        costOut.append(prn.NNOut(netIn, algorithm.nets[i]).squeeze())

    costOut = np.array(costOut)
    costOut = costOut.mean(axis = 0)

    return costOut

if __name__ == "__main__":
    # declare CFDNNetAdapt
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

    # algorithm parameters
    algorithm.nSam = 1000
    algorithm.deltaNSams = [1000]
    algorithm.nNN = 4
    algorithm.minN = 2
    algorithm.maxN = 20
    algorithm.nHidLay = 3
    algorithm.tol = 1e-5
    algorithm.iMax = 200
    algorithm.dRN = 1
    algorithm.nComps = 16
    algorithm.nSeeds = 4

    # parameters for ANN training
    algorithm.trainPro = 75
    algorithm.valPro = 15
    algorithm.testPro = 10
    algorithm.kMax = 10000
    algorithm.rEStop = 1e-5
    algorithm.dnnVerbose = False

    # parameters for MOP
    algorithm.pMin = 0.0
    algorithm.pMax = 1.0
    algorithm.offSize = 100
    algorithm.popSize = 100
    algorithm.nGens = 250

    # evaluation funcs
    algorithm.dnnEvalFunc = dnnEvaluation
    algorithm.smpEvalFunc = smpEvaluation

    # initialize
    algorithm.initialize()

    # run
    algorithm.run()
