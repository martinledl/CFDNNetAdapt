
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
from CFDNNetAdapt import *
from postProcesser import *
from configureAndRun import *

# CFD simulation settings
cwd = os.getcwd() + "/"
genDir = cwd
baseDir = cwd + "ZZ_cases/"
baseCase = genDir + "10_baseCase/"
cleanSims = True

# things to change in caseConstructor
consPars = ["QInLst", "pSucLst", "DNz2", "endTime", "edgeFunction", "genDir", "baseCase", "baseDir", "LMxT", "convCPs", "diffuserCPs", "LConv", "LDiff"]

QInLst = [0.3e-3, 0.4e-3, 0.5e-3]
pSucLst = [81.325, 81.325, 81.325]

DNz2 = 0.0043
maxLength = 1.0

endTime = 5000

edgeFunction = "polyLine"
nCPConv = 2
nCPDiff = 4

caseConstructor = "caseConstructor.py"
Allrun = "Allrun-serialAll"

resField = "p"
resThres = 1e-3

# fixed geometric parameters
WConv = 0.0175
WMxT = 0.014*0.5
WDiff = 0.0178
LNz123 = 0.06
LMxT = 0.2
LGap = 0.015

# default parameters
defaultParameters = [0.3, 0.3, 0.6, 0.6, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 0.02, 0.13]

# fixed costFunctions parameters
xInl = 0.005
xSuc = 0.005
rho = 1000.0
hCut = 0.01

maxOrMin = [-1, 1] # whether to maximize (-1) or minimize (1) the objective

# bounds
xMin = 0.0
xMax = 1.0
yMin = 0.0
yMax = 2.0

lenMin = 0.01
lenMax = 1.0-lenMin-LGap-LMxT

# notes for simulation run
noteDict = {
    "finished successfully":0,
    "strange values":1,
    "bad mesh":2,
    "did not converge":3,
    "not computed":4,
    "bad parameters":5,
    "value not found":6,
}

# cfd evaluation func
def cfdEvaluation(i):
    caseID = os.getpid()

    # evaluate the cases
    cfdOut = [0] * (algorithm.nObjs)
    isBad = False
    note = 0

    netPars = algorithm.result[i].variables[:]
    netOuts = algorithm.result[i].objectives[:]

    # rescale the pars
    for p in range(len(netPars)):
        netPars[p] = netPars[p]*(algorithm.cfdMaxs[p] - algorithm.cfdMins[p]) + algorithm.cfdMins[p]

    # assign pars
    convCPs = [[netPars[0], netPars[1]], [netPars[2], netPars[3]]]
    diffuserCPs = [[netPars[4], netPars[5]],[netPars[6], netPars[7]],[netPars[8],netPars[9]],[netPars[10],netPars[11]]]
    LConv = netPars[-2]
    LDiff = netPars[-1]

    # check parameters acceptance
    badPars = False
    for j in range(nCPConv-1):
        if convCPs[j][0] >= convCPs[j+1][0]:
            badPars = True

    for j in range(nCPDiff-1):
        if diffuserCPs[j][0] >= diffuserCPs[j+1][0]:
            badPars = True

    if LGap+LConv+LMxT+LDiff > maxLength:
        badPars = True

    if badPars:
        note = noteDict.get("bad parameters")

    else:
        # create and compute the simulation
        caseConsArgs = []
        pars = consPars[:]
        parValues = []
        for par in pars:
            parValues.append(eval(par))

        changeDict = {
                "changeParsInCaseConstructor":[pars, parValues]
                }

        # redefine costFunctions with respect to currect LDiff
        xRef = LNz123+LGap+LConv+LMxT-0.005
        xOut = LNz123+LGap+LConv+LMxT+LDiff
        costFunctionDict = {
            "calcEnergyEfficiencyV1 " + str(xInl) + " " + str(xSuc) + " " + str(xOut) + " " + str(rho) + " " + str(hCut):"Eeff = ",
            }

        costKeys = costFunctionDict.keys()

        simulations = configureAndRun(genDir, caseConstructor, baseCase, caseID = caseID)
        simulations.makeCase(changeDict) # create simulation

        caseDirList = glob.glob(baseDir+"*"+str(caseID))

        for caseDir in caseDirList:
            simulations.writePostProcessing(costKeys, caseDir + "/", Allrun) # write cost functions to Allrun
            simulations.runSingleCase(caseDir + "/", Allrun) # run simulation

        # evaluate the cases
        for caseDir in caseDirList:
            case = postProcesser(caseDir + "/")
            if not case.meshCheck(checkMesh = False): # checkMesh checked during simulation evaluation
                note = noteDict.get("bad mesh")
                break

            if not case.isComputed(): # checkMesh check will show itself here
                note = noteDict.get("not computed")
                break

            if not case.finalResidualCheck(resField, resThres):
                note = noteDict.get("did not converge")
                break

            for cID, costFunction in enumerate(costKeys):
                logName = "log." + costFunction.split()[0]
                idStr = costFunctionDict.get(costFunction)

                found = case.findInLog(logName, idStr)
                if type(found) == str:
                    value = float(found.split(idStr)[-1])
                    S = maxOrMin[cID]*value

                    if S <= -1:
                        note = noteDict.get("strange values")

                    cfdOut[cID] += S

                else:
                    note = noteDict.get("value not found")

                # evaluate total length
                cfdOut[cID+1] += LConv+LDiff

        # clean simulation folder
        if cleanSims:
            for caseDir in caseDirList:
                sh.rmtree(caseDir)

    # rescale for ANN
    cfdOut = [o/len(QInLst) for o in cfdOut]
    for co in range(len(cfdOut)):
        cfdOut[co] = (cfdOut[co] - algorithm.cfdMins[algorithm.nPars+co])/(algorithm.cfdMaxs[algorithm.nPars+co] - algorithm.cfdMins[algorithm.nPars+co])

    # check notes
    if not note == 0:
        isBad = True

    # compute the error
    delta = 0
    if not isBad:
        delta += abs(netOuts[0] - cfdOut[0])
        delta += abs(netOuts[1] - cfdOut[1])

    else:
        delta = -1

    algorithm.outFile.write("Doing CFD check no. " + str(algorithm.toCompare.index(i)) + " with parameters " + str(netPars) + "\n")
    algorithm.outFile.write("CFD done with note " + str(note) + "\n")
    algorithm.outFile.write("no. " + str(algorithm.toCompare.index(i)) + " ANN outs were " + str(netOuts) + "\n")
    algorithm.outFile.write("no. " + str(algorithm.toCompare.index(i)) + " CFD outs were " + str(cfdOut) + " delta " + str(delta) + "\n")
    algorithm.outFile.write("CFD check no. " + str(algorithm.toCompare.index(i)) + " done\n")
    algorithm.outFile.flush()

    return delta

def dnnEvaluation(vars):
    """ function to return the costs for optimization """
    dummy = 1e6

    # rescale the pars
    netPars = list()
    for p in range(len(vars)):
        netPars.append(vars[p]*(algorithm.cfdMaxs[p] - algorithm.cfdMins[p]) + algorithm.cfdMins[p])

    convCPs = [[netPars[0], netPars[1]], [netPars[2], netPars[3]]]
    diffuserCPs = [[netPars[4], netPars[5]],[netPars[6], netPars[7]],[netPars[8],netPars[9]],[netPars[10],netPars[11]]]
    LConv = netPars[-2]
    LDiff = netPars[-1]

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

    if isGood:
        netIn = np.array(vars)
        netIn = np.expand_dims(netIn, axis = 1)

        costOut = list()

        for i in range(len(algorithm.nets)):
            costOut.append(prn.NNOut(netIn, algorithm.nets[i]).squeeze())

        costOut = np.array(costOut)
        costOut = costOut.mean(axis = 0)

        totLen = LDiff+LConv
        totLen = (totLen - algorithm.cfdMins[-1])/(algorithm.cfdMaxs[-1] - algorithm.cfdMins[-1])
        return [costOut,totLen]

    else:
        return np.array([dummy]*(algorithm.nObjs))

# declare CFDNNetAdapt
algorithm = CFDNNetAdapt()

# problem specification
algorithm.nPars = 14
algorithm.nObjs = 2
algorithm.nOuts = 1
algorithm.mainDir = "01_algoRuns/"
algorithm.cfdDir = "00_prepCFDData/"
algorithm.prbDir = "14_LConv2CPLDiff4CP/"
algorithm.dataNm = "10_platypusCFDAllSolutions.dat"
algorithm.minMax = "12_minMaxAng.dat"

# algorithm parameters
algorithm.nSam = 2000
algorithm.deltaNSam = 500
algorithm.nNN = 10
algorithm.minN = 2
algorithm.maxN = 20
algorithm.nHidLay = 3
algorithm.tol = 5e-2
algorithm.iMax = 200
algorithm.dRN = 0
algorithm.nComps = 8
algorithm.nSeeds = 5

# parameters for ANN training
algorithm.trainPro = 75
algorithm.valPro = 15
algorithm.testPro = 10
algorithm.kMax = 10000
algorithm.rEStop = 1e-5

# parameters for MOP
algorithm.pMin = 0.0
algorithm.pMax = 1.0
algorithm.offSize = 500
algorithm.popSize = 500
algorithm.nGens = 30

# evaluation funcs
algorithm.dnnEvalFunc = dnnEvaluation
algorithm.cfdEvalFunc = cfdEvaluation

# initialize
algorithm.initialize()

# define constrains
constr = [maxLength - LGap - LMxT]
with open(algorithm.cfdDir + algorithm.prbDir + algorithm.minMax, 'r') as file:
    reader = csv.reader(file)

    cols = next(reader)

    for line in reader:
        constr.append(float(line[0]))
        constr.append(float(line[1]))

# run
algorithm.run()
