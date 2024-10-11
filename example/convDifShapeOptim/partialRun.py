
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
from CFDNNetAdaptV4 import *

# geometric parameters
maxLength = 1.0
WConv = 0.0175
WMxT = 0.014*0.5
WDiff = 0.0178
LMxT = 0.2
LGap = 0.015

# sample evaluation function
def smpEvaluation(i):
    """ should not be required """
    return True

# dnn evaluation function
def dnnEvaluation(vars):
    """ function to return the costs for optimization """
    dummy = 1e6

    # rescale parameters
    netPars = list()
    for p in range(len(vars)):
        netPars.append(vars[p]*(algorithm.smpMaxs[p] - algorithm.smpMins[p]) + algorithm.smpMins[p])

    # sort parameters
    convCPs = [[netPars[0], netPars[1]], [netPars[2], netPars[3]]]
    diffuserCPs = [[netPars[4], netPars[5]],[netPars[6], netPars[7]],[netPars[8],netPars[9]],[netPars[10],netPars[11]]]
    LConv = netPars[-2]
    LDiff = netPars[-1]

    # check constraints
    isGood = True
    for j in range(len(convCPs)-1):
        if convCPs[j][0] >= convCPs[j+1][0]:
            isGood = False

    for j in range(len(diffuserCPs)-1):
        if diffuserCPs[j][0] >= diffuserCPs[j+1][0]:
            isGood = False

    if LConv+LDiff > constr[0]:
        isGood = False

    # check angles
    xC1 = netPars[0]
    yC1 = netPars[1]
    xC2 = netPars[2]
    yC2 = netPars[3]
    xD1 = netPars[4]
    yD1 = netPars[5]
    xD2 = netPars[6]
    yD2 = netPars[7]
    xD3 = netPars[8]
    yD3 = netPars[9]
    xD4 = netPars[10]
    yD4 = netPars[11]
    LConv = netPars[12]
    LDiff = netPars[13]

    # converging part
    dXC1 = (xC1 - 0)*LConv
    dXC2 = (xC2 - xC1)*LConv
    dXC3 = (1 - xC2)*LConv

    dYC1 = (1 - yC1)*(WConv - WMxT)
    dYC2 = (yC1 - yC2)*(WConv - WMxT)
    dYC3 = (yC2 - 0)*(WConv - WMxT)

    angC1 = math.atan(dYC1/dXC1)
    if angC1 < constr[1] or angC1 > constr[2]:
        isGood = False
    angC2 = math.atan(dYC2/dXC2)
    if angC2 < constr[3] or angC2 > constr[4]:
        isGood = False
    angC3 = math.atan(dYC3/dXC3)
    if angC3 < constr[5] or angC3 > constr[6]:
        isGood = False

    # diffuser
    dXD1 = (xD1 - 0)*LDiff
    dXD2 = (xD2 - xD1)*LDiff
    dXD3 = (xD3 - xD2)*LDiff
    dXD4 = (xD4 - xD3)*LDiff
    dXD5 = (1 - xD4)*LDiff

    dYD1 = (yD1 - 0)*(WDiff - WMxT)
    dYD2 = (yD2 - yD1)*(WDiff - WMxT)
    dYD3 = (yD3 - yD2)*(WDiff - WMxT)
    dYD4 = (yD4 - yD3)*(WDiff - WMxT)
    dYD5 = (1 - yD4)*(WDiff - WMxT)

    angD1 = math.atan(dYD1/dXD1)
    if angD1 < constr[7] or angD1 > constr[8]:
        isGood = False
    angD2 = math.atan(dYD2/dXD2)
    if angD2 < constr[9] or angD2 > constr[10]:
        isGood = False
    angD3 = math.atan(dYD3/dXD3)
    if angD3 < constr[11] or angD3 > constr[12]:
        isGood = False
    angD4 = math.atan(dYD4/dXD4)
    if angD4 < constr[13] or angD4 > constr[14]:
        isGood = False
    angD5 = math.atan(dYD5/dXD5)
    if angD5 < constr[15] or angD5 > constr[16]:
        isGood = False

    # is constraints satisfied
    if isGood:
        netIn = np.array(vars)
        netIn = np.expand_dims(netIn, axis = 1)

        costOut = list()

        for i in range(len(algorithm.nets)):
            costOut.append(prn.NNOut(netIn, algorithm.nets[i]).squeeze())

        costOut = np.array(costOut)
        costOut = costOut.mean(axis = 0)

        totLen = LDiff+LConv
        totLen = (totLen - algorithm.smpMins[-1])/(algorithm.smpMaxs[-1] - algorithm.smpMins[-1])
        return [costOut,totLen]

    # if not return big dummy value
    else:
        return np.array([dummy]*(algorithm.nObjs))

# declare CFDNNetAdapt
algorithm = CFDNNetAdapt()

# problem specification
algorithm.nPars = 14
algorithm.nObjs = 2
algorithm.nOuts = 1
algorithm.mainDir = "01_algoRuns/"
algorithm.smpDir = "00_prepCFDData/"
algorithm.prbDir = "14_LConv2CPLDiff4CP/"
algorithm.dataNm = "10_platypusCFDAllSolutions.dat"
algorithm.minMax = "12_minMaxAng.dat"

# algorithm parameters
algorithm.nSam = 2000
algorithm.deltaNSams = [2000]
algorithm.nNN = 1
algorithm.minN = 5
algorithm.maxN = 20
algorithm.nHidLay = 3
algorithm.tol = 5e-2
algorithm.iMax = 4
algorithm.dRN = 0
algorithm.nComps = 0 # smpEvaluation not used
algorithm.nSeeds = 1

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
algorithm.offSize = 500
algorithm.popSize = 500
algorithm.nGens = 30

# evaluation funcs
algorithm.dnnEvalFunc = dnnEvaluation
algorithm.smpEvalFunc = smpEvaluation

# initialize
algorithm.initialize()

# define constrains
constr = [maxLength - LGap - LMxT]
with open(algorithm.smpDir + algorithm.prbDir + algorithm.minMax, 'r') as file:
    reader = csv.reader(file)

    cols = next(reader)

    for line in reader:
        constr.append(float(line[0]))
        constr.append(float(line[1]))

# run
algorithm.run()
