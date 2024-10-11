import numpy as np
import math
import csv

import levenberg_marquardt as lm
from tensorflow.keras.optimizers.legacy import SGD
from testFunctions import ZDT6


def dnnEvaluation(args, nets, lm_optimizer=False, **kwargs):
    """ function to return the costs for optimization """

    netIn = np.array(args)
    netIn = np.expand_dims(netIn, axis=0)
    # netIn = netIn.reshape((-1, 2))

    costOut = list()

    for i in range(len(nets)):
        model = nets[i]

        if lm_optimizer:
            model = lm.ModelWrapper(model)
            model.compile(
                optimizer=SGD(learning_rate=1.0),
                loss=lm.MeanSquaredError())

        output = model(netIn, training=False)
        costOut.append(np.reshape(output, (2,)))

    costOut = np.array(costOut)
    costOut = costOut.mean(axis=0)

    return costOut


def smpEvaluation(args, **kwargs):
    population, i, smpMins, smpMaxs, nPars = args
    netPars = population[i].variables[:]
    netOuts = population[i].objectives[:]

    # rescale the pars
    for p in range(len(netPars)):
        netPars[p] = netPars[p] * (smpMaxs[p] - smpMins[p]) + smpMins[p]

    checkOut = ZDT6(netPars)

    # rescale for ANN
    for co in range(len(checkOut)):
        checkOut[co] = (checkOut[co] - smpMins[nPars + co]) / (
                smpMaxs[nPars + co] - smpMins[nPars + co])

    # compute the error
    delta = 0
    delta += abs(netOuts[0] - checkOut[0])
    delta += abs(netOuts[1] - checkOut[1])

    return delta


# sample evaluation function
def smpEvaluation2(i):
    """ should not be required """
    return True


# dnn evaluation function
def dnnEvaluation2(args, nets, smpMins, smpMaxs, nObjs, nOuts, smpDir, prbDir, minMax, lm_optimizer):
    """ function to return the costs for optimization """
    # geometric parameters
    maxLength = 1.0
    WConv = 0.0175
    WMxT = 0.014 * 0.5
    WDiff = 0.0178
    LMxT = 0.2
    LGap = 0.015

    # define constrains
    constr = [maxLength - LGap - LMxT]
    with open(smpDir + prbDir + minMax, 'r') as file:
        reader = csv.reader(file)

        cols = next(reader)

        for line in reader:
            constr.append(float(line[0]))
            constr.append(float(line[1]))

    dummy = 1e6

    # rescale parameters
    netPars = list()
    for p in range(len(args)):
        netPars.append(args[p] * (smpMaxs[p] - smpMins[p]) + smpMins[p])

    # sort parameters
    convCPs = [[netPars[0], netPars[1]], [netPars[2], netPars[3]]]
    diffuserCPs = [[netPars[4], netPars[5]], [netPars[6], netPars[7]], [netPars[8], netPars[9]],
                   [netPars[10], netPars[11]]]
    LConv = netPars[-2]
    LDiff = netPars[-1]

    # check constraints
    isGood = True
    for j in range(len(convCPs) - 1):
        if convCPs[j][0] >= convCPs[j + 1][0]:
            isGood = False

    for j in range(len(diffuserCPs) - 1):
        if diffuserCPs[j][0] >= diffuserCPs[j + 1][0]:
            isGood = False

    if LConv + LDiff > constr[0]:
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
    dXC1 = (xC1 - 0) * LConv
    dXC2 = (xC2 - xC1) * LConv
    dXC3 = (1 - xC2) * LConv

    dYC1 = (1 - yC1) * (WConv - WMxT)
    dYC2 = (yC1 - yC2) * (WConv - WMxT)
    dYC3 = (yC2 - 0) * (WConv - WMxT)

    angC1 = math.atan(dYC1 / dXC1)
    if angC1 < constr[1] or angC1 > constr[2]:
        isGood = False
    angC2 = math.atan(dYC2 / dXC2)
    if angC2 < constr[3] or angC2 > constr[4]:
        isGood = False
    angC3 = math.atan(dYC3 / dXC3)
    if angC3 < constr[5] or angC3 > constr[6]:
        isGood = False

    # diffuser
    dXD1 = (xD1 - 0) * LDiff
    dXD2 = (xD2 - xD1) * LDiff
    dXD3 = (xD3 - xD2) * LDiff
    dXD4 = (xD4 - xD3) * LDiff
    dXD5 = (1 - xD4) * LDiff

    dYD1 = (yD1 - 0) * (WDiff - WMxT)
    dYD2 = (yD2 - yD1) * (WDiff - WMxT)
    dYD3 = (yD3 - yD2) * (WDiff - WMxT)
    dYD4 = (yD4 - yD3) * (WDiff - WMxT)
    dYD5 = (1 - yD4) * (WDiff - WMxT)

    angD1 = math.atan(dYD1 / dXD1)
    if angD1 < constr[7] or angD1 > constr[8]:
        isGood = False
    angD2 = math.atan(dYD2 / dXD2)
    if angD2 < constr[9] or angD2 > constr[10]:
        isGood = False
    angD3 = math.atan(dYD3 / dXD3)
    if angD3 < constr[11] or angD3 > constr[12]:
        isGood = False
    angD4 = math.atan(dYD4 / dXD4)
    if angD4 < constr[13] or angD4 > constr[14]:
        isGood = False
    angD5 = math.atan(dYD5 / dXD5)
    if angD5 < constr[15] or angD5 > constr[16]:
        isGood = False

    # is constraints satisfied
    if isGood:
        netIn = np.array(args)
        netIn = np.expand_dims(netIn, axis=0)

        costOut = list()

        for i in range(len(nets)):
            model = nets[i]

            if lm_optimizer:
                model = lm.ModelWrapper(model)
                model.compile(
                    optimizer=SGD(learning_rate=1.0),
                    loss=lm.MeanSquaredError())

            output = model(netIn, training=False)
            costOut.append(np.reshape(output, (nOuts,)))

        costOut = np.array(costOut)
        # -> [[cost]]
        costOut = costOut.mean(axis=0).mean(axis=0)
        # -> cost

        totLen = LDiff + LConv
        totLen = (totLen - smpMins[-1]) / (smpMaxs[-1] - smpMins[-1])
        return [costOut, totLen]

    # if not return big dummy value
    else:
        return np.array([dummy] * nObjs)