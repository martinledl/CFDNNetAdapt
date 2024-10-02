from CFDNNetAdaptV4 import CFDNNetAdapt


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
    algorithm.nSam = 2000
    algorithm.deltaNSams = [2000]
    algorithm.nNN = 1
    algorithm.minN = 5
    algorithm.maxN = 20
    algorithm.nHidLay = 3
    algorithm.tol = 1e-5
    algorithm.iMax = 3
    algorithm.dRN = 1
    algorithm.nComps = 16
    algorithm.nSeeds = 1

    # parameters for ANN training
    algorithm.trainPro = 75
    algorithm.valPro = 15
    algorithm.testPro = 10
    algorithm.kMax = 100
    algorithm.rEStop = 1e-5
    algorithm.activationFunction = "tanh"
    algorithm.verbose = True
    algorithm.nValFails = 150
    algorithm.batchSize = 256
    algorithm.drawTrainingPlot = True

    # parameters for MOP
    algorithm.pMin = 0.0
    algorithm.pMax = 1.0
    algorithm.offSize = 100
    algorithm.popSize = 100
    algorithm.nGens = 250

    # initialize
    algorithm.initialize()

    # run
    algorithm.run()
