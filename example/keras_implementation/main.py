import os
import sys
import csv
import math
import numpy as np
import dill as pickle
import multiprocessing
from CFDNNetAdaptV4 import CFDNNetAdapt

sys.path.insert(1, "../../src")
sys.path.insert(1, "../../thirdParty")

# bounds
pMin = 0.0
pMax = 1.0


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
    algorithm.nSam = 4000
    algorithm.deltaNSams = [4000]
    algorithm.nNN = 1
    algorithm.minN = 2
    algorithm.maxN = 4
    algorithm.nHidLay = 3
    algorithm.tol = 1e-5
    algorithm.iMax = 4
    algorithm.dRN = 1
    algorithm.nComps = 1
    algorithm.nSeeds = 1

    # parameters for ANN training
    algorithm.trainPro = 75
    algorithm.valPro = 15
    algorithm.testPro = 10
    algorithm.kMax = 5
    algorithm.rEStop = 1e-2
    algorithm.dnnVerbose = True

    # parameters for MOP
    algorithm.pMin = 0.0
    algorithm.pMax = 1.0
    algorithm.offSize = 10
    algorithm.popSize = 10
    algorithm.nGens = 2

    # evaluation funcs
    # algorithm.dnnEvalFunc = dnnEvaluation
    # algorithm.smpEvalFunc = smpEvaluation

    # initialize
    algorithm.initialize()

    # run
    algorithm.run()
