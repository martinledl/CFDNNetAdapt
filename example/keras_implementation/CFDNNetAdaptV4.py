import os
import sys
import math
import random
import datetime
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import multiprocessing

sys.path.insert(1, "../../thirdParty")
import platypusModV2 as plat

sys.path.insert(1, "../../src")
from testFunctions import ZDT6


class CFDNNetAdapt:
    def __init__(self):
        # optimization problem
        self.nPars = None  # number of parameters
        self.nObjs = None  # number of objectives
        self.nOuts = None  # number of networks outputs

        # CFDNNetAdapt hyperparameters
        self.nSam = None  # initial number of samples
        self.deltaNSams = None  # factor to change number of samples, may be a list to change among iterations
        self.nNN = None  # number of neural networks to test
        self.tol = None  # required tolerance
        self.iMax = None  # maximum number of iterations
        self.dRN = None  # factor to change variance of random number of neurons selection
        self.nComps = None  # number of verification checks
        self.nSeeds = None  # number of seeds
        self.nets = None

        # DNN parameters
        self.minN = None  # minimal number of neurons
        self.maxN = None  # maximal number of neurons
        self.nHidLay = None  # number of hidden layer
        self.trainPro = None  # percentage of samples used for training
        self.valPro = None  # percentage of samples used for validation
        self.testPro = None  # percentage of samples used for testing
        self.kMax = None  # maximum number of iterations for dnn training
        self.rEStop = None  # required error for dnn validation
        self.dnnVerbose = False  # print info about dnn training

        # MOEA parameters
        self.pMin = None  # minimal allowed parameter value
        self.pMax = None  # maximal allowed parameter value
        self.offSize = None  # offspring size
        self.popSize = None  # population size
        self.nGens = None  # number of generations

        # directories and data files
        self.mainDir = None  # main save directory
        self.smpDir = None  # directory with samples
        self.prbDir = None  # specific data directory
        self.dataNm = None  # name of the file with data
        self.specRunDir = None  # specified run directory, optional

        # evaluation functions
        # self.dnnEvalFunc = None  # custom function for dnn evaluation in optimization
        # self.smpEvalFunc = None  # custom function for sample evaluation in verification

        # flags
        self.toPlotReg = False  # wheter to create regression plots, requires uncommenting matplotlib import

        # internal variables
        self.verbosityLevel = 0  # verbosity level for Keras


    def initialize(self):
        # prepare DNN specifics
        self.netTransfer = ["tanh"] * self.nHidLay  # transfer functions
        self.nValFails = self.nHidLay * 10  # allowed number of failed validations
        self.nHid = [(self.maxN + self.minN) / 2 for _ in range(self.nHidLay)]  # mean number of neurons for each layer
        self.rN = (self.maxN - self.minN) / 2  # variance for random number of neurons selection
        self.rN *= 0.5

        # calculate internal variables
        self.verbosityLevel = 1 if self.dnnVerbose else 0

        # prepare directories
        self.prepOutDir(self.mainDir)
        if self.specRunDir is None:
            ls = os.listdir(self.mainDir)
            ls = [i for i in ls if "run" in i]
            self.runDir = self.mainDir + f"run_{(len(ls) + 1):02d}/"
        else:
            self.runDir = self.mainDir + self.specRunDir
        self.prepOutDir(self.runDir)

        # prepare mins and maxs for scaling
        self.smpMins, self.smpMaxs = self.getScalesFromFile(self.smpDir + self.prbDir, self.dataNm)

        # prepare samples
        self.source, self.target = self.loadAndScaleData(self.smpDir + self.prbDir, self.dataNm, self.nPars, self.nOuts)
        self.souall, self.tarall = self.loadAndScaleData(self.smpDir + self.prbDir, self.dataNm, self.nPars, self.nObjs)
        self.maxSam = np.shape(self.source)[1]  # maximum number of samples

    def createNN(self, stepDir):
        newCheck = True
        skip = False
        netTry = 1

        while newCheck:
            nMins = [max(int(self.nHid[i] - self.rN), self.minN) for i in range(self.nHidLay)]
            nMaxs = [min(int(self.nHid[i] + self.rN), self.maxN) for i in range(self.nHidLay)]

            netStruct = [self.nPars]
            for i in range(self.nHidLay):
                netStruct += [random.randint(nMins[i], nMaxs[i])]
            netStruct += [self.nOuts]

            netNm = "_".join([str(i) for i in netStruct])
            netDir = stepDir + netNm + "/"

            if not os.path.exists(netDir):
                newCheck = False
            elif netTry >= self.nNN:
                newCheck = False
                skip = True

            netTry += 1

        return netStruct, netNm, netDir, skip

    def createRandomDNNs(self, stepDir):
        netStructs = []
        netNms = []
        netDirs = []

        for _ in range(self.nNN):
            netStruct, netNm, netDir, skip = self.createNN(stepDir)
            if skip:
                continue

            self.prepOutDir(netDir)
            self.writeToLog("Created net " + str(netNm) + "\n")

            netStructs.append(netStruct)
            netNms.append(netNm)
            netDirs.append(netDir)

        return netStructs, netNms, netDirs

    @staticmethod
    def build_model(netStruct, activations):
        model = Sequential()

        # First layer manages the input shape
        model.add(InputLayer(shape=(netStruct[0],)))

        # Add hidden layers
        for (neurons, activation) in zip(netStruct[1:-1], activations):
            model.add(Dense(neurons, activation=activation))

        # Add output layer
        model.add(Dense(netStruct[-1], activation='linear'))

        model.compile(optimizer=Adam(), loss='mean_squared_error')
        return model

    def train_model(self, model, sourceTr, targetTr, sourceVl, targetVl):
        # Stops the training in case the validation loss is getting higher and revert to the best weights before this
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.nValFails, restore_best_weights=True)
        history = model.fit(sourceTr.T, targetTr.T, validation_data=(sourceVl.T, targetVl.T), epochs=self.kMax,
                            verbose=self.verbosityLevel, callbacks=[early_stopping], batch_size=256)
        return history

    def dnnSeedEvaluation(self, args):
        # unpack arguments
        netStruct, netTransfer, sourceTr, targetTr, sourceVl, targetVl, sourceTe, targetTe, kMax, rEStop, nValFails, dnnVerbose, runDir, iteration, seed = args

        model = self.build_model(netStruct, netTransfer)
        self.train_model(model, sourceTr, targetTr, sourceVl, targetVl)

        stepDir = runDir + f"step_{iteration:04d}/"
        netNm = "_".join([str(i) for i in netStruct])
        netDir = stepDir + netNm + "/"
        model.save(netDir + f'{netNm}_{seed:03d}.keras')

        loss = model.evaluate(sourceTe.T, targetTe.T, verbose=self.verbosityLevel)

        return loss

    def packDataForDNNTraining(self, netStructs):
        # pack arguments for parallel evaluation of dnnSeedEvaluation function
        arguments = list()
        for n in range(len(netStructs)):
            for i in range(self.nSeeds):
                argument = ([netStructs[n],  # network architectures
                             ["tanh"] * self.nHidLay,  # transfer functions
                             self.sourceTr, self.targetTr,  # training samples
                             self.sourceVl, self.targetVl,  # validatioin samples
                             self.sourceTe, self.targetTe,  # testing samples
                             self.kMax, self.rEStop,  # maximum number of iterations and required training error
                             self.nValFails, self.dnnVerbose,  # number of allowed validation failes and verbose flag
                             self.runDir, self.iteration,  # save directory and iteration counter
                             i])  # parallel counter
                arguments.append(argument)

        return arguments

    def run(self):
        self.startLog()
        last = False
        epsilon = 1
        prevSamTotal = 0
        smpNondoms = None
        self.iteration = 1

        while epsilon > self.tol and self.iteration <= self.iMax:
            stepDir = self.runDir + f"step_{self.iteration:04d}/"
            self.prepOutDir(stepDir)
            # log
            self.writeToLog("Starting iteration " + str(self.iteration) + "\n")

            # compute number of samples used
            nSamTotal, trainLen, valLen, testLen = self.prepareSamples()

            # find pareto front from samples
            smpNondoms = self.getNondomSolutionsFromSamples(prevSamTotal, nSamTotal, smpNondoms)

            self.writeToLog(f"Using {self.nSam} training samples, {valLen} validation samples and {testLen} testSamples\n")

            # check the last best dnn
            if self.iteration > 1:
                self.checkLastBestDNN(netNondoms, smpNondoms)

            netStructs, netNms, netDirs = self.createRandomDNNs(stepDir)
            arguments = self.packDataForDNNTraining(netStructs)

            parallelNum = self.nSeeds * len(netStructs)

            with multiprocessing.Pool(parallelNum) as p:
                cErrors = p.map(self.dnnSeedEvaluation, arguments)
            # cErrors = []
            # for i in range(self.nSeeds * len(netStructs)):
            #     cError = self.dnnSeedEvaluation(arguments[i])
            #     if cError is not None:
            #         print(f"NN {i} loss: {cError:.4f}")
            #     cErrors.append(cError)

            if self.toPlotReg:
                self.plotRegressionGraph(netStructs, netNms, netDirs)

            self.writeToLog(f"Iteration {self.iteration} - Training finished\n")

            bestNet, netNondoms = self.optimizeAndFindBestDNN(netStructs, netNms, netDirs, smpNondoms)
            self.writeToLog(f"Iteration {self.iteration} - Best DNN found {bestNet}\n")

            delta, bads = self.runVerification(bestNet, stepDir)

            if bads == self.nComps:
                if (self.iteration - 1) >= len(self.deltaNSams):
                    deltaNSam = self.deltaNSams[-1]
                else:
                    deltaNSam = self.deltaNSams[self.iteration - 1]

                self.nSam -= deltaNSam
                self.rN += self.dRN
            else:
                epsilon = delta / (self.nComps - bads)

            self.writeToLog(f"Last residual - {epsilon}\n\n")

            if last:
                self.writeToLog("Done. Maximum number of samples reached\n")
                self.finishLog()
                exit()

            prevSamTotal, nSamTotal, last = self.prepareForNextIter(bestNet, prevSamTotal, nSamTotal)

        self.writeToLog("Done. Required error reached\n")
        self.finishLog()

    def writeToLog(self, text):
        with open(self.runDir + "log.out", 'a') as outFile:
            outFile.write(text)

    def startLog(self):
        self.writeToLog("\nstartTime = " + str(datetime.datetime.now().strftime("%d/%m/%Y %X")) + "\n")
        self.writeToLog("===================SET UP=====================\n")

        # prepare things to write
        toWrite = [
            "nPars", "nOuts", "nObjs",
            "nSam", "deltaNSams",
            "nNN", "minN", "maxN", "nHidLay",
            "tol", "iMax", "dRN",
            "nComps", "nSeeds",
            "trainPro", "valPro", "testPro",
            "kMax", "rEStop", "nValFails",
            "pMin", "pMax",
            "offSize", "popSize", "nGens"]

        # write
        for thing in toWrite:
            self.writeToLog(f"{thing} = {eval('self.' + thing)}\n")

        # finish
        self.writeToLog("\n")


    def finishLog(self):
        # write ending and close
        self.writeToLog("==============================================\n")
        self.writeToLog(f"endTime = {datetime.datetime.now().strftime('%d/%m/%Y %X')}\n")

    def prepareSamples(self):
        # total number of samples used in iteration
        nSamTotal = int(self.nSam / self.trainPro * 100)

        # take part of samples
        cSource = self.source[:, :nSamTotal]
        cTarget = self.target[:, :nSamTotal]

        # get training, validation and testing lengths
        trainLen = int(self.trainPro / 100 * nSamTotal)
        valLen = int(self.valPro / 100 * nSamTotal)
        testLen = nSamTotal - trainLen - valLen

        # sort samples
        self.sourceTr = cSource[:, :trainLen]
        self.targetTr = cTarget[:, :trainLen]

        self.sourceVl = cSource[:, trainLen:trainLen + valLen]
        self.targetVl = cTarget[:, trainLen:trainLen + valLen]

        self.sourceTe = cSource[:, trainLen + valLen:]
        self.targetTe = cTarget[:, trainLen + valLen:]

        return nSamTotal, trainLen, valLen, testLen

    def getNondomSolutionsFromSamples(self, prevSamTotal, nSamTotal, smpNondoms=None):
        # get samples added in this iteration
        aSource = self.souall[:, :nSamTotal]
        aTarget = self.tarall[:, :nSamTotal]

        # concatenate with last iteration nondominated solutions
        aAll = np.append(aSource, aTarget, axis=0)
        if self.iteration > 1:
            aAll = np.concatenate((smpNondoms.T, aAll), axis=1)

        # find current nondominated solutions
        nondoms = self.findNondominatedSolutions(aAll.T, [1, 1])
        return nondoms

    def checkLastBestDNN(self, netNondoms, smpNondoms):
        # compare pareto fronts
        dists = self.compareParetoFronts(netNondoms, smpNondoms)

        # compute and write total error
        pError = sum(dists) / len(dists)
        self.writeToLog(f"Error of best DNN from last iteration is {pError}\n")

        # end run if error small enough
        if pError < self.tol:
            self.writeToLog(f"Done. Last best DNN error < {self.tol}\n")
            self.finishLog()
            exit()

    @staticmethod
    def dnnEvaluation(vars):
        """ function to return the costs for optimization """

        netIn = np.array(vars)
        netIn = np.expand_dims(netIn, axis=0)
        # netIn = netIn.reshape((-1, 2))

        costOut = list()

        mainDir = "01_algoRuns/"
        runDir = os.path.join(mainDir, sorted(os.listdir(mainDir))[-1])
        stepDir = os.path.join(runDir, sorted([d for d in os.listdir(runDir) if "step_" in d])[-1])
        nets = []
        for netDir in os.listdir(stepDir):
            for file in os.listdir(os.path.join(stepDir, netDir)):
                if file.endswith(".keras"):
                    nets.append(load_model(os.path.join(stepDir, netDir, file)))

        for i in range(len(nets)):
            model = nets[i]
            output = model(netIn, training=False)
            costOut.append(np.reshape(output, (2,)))

        costOut = np.array(costOut)
        costOut = costOut.mean(axis=0)

        return costOut

    # cost function evaluation
    def smpEvaluation(self, i):
        netPars = self.population[i].variables[:]
        netOuts = self.population[i].objectives[:]

        # rescale the pars
        for p in range(len(netPars)):
            netPars[p] = netPars[p] * (self.smpMaxs[p] - self.smpMins[p]) + self.smpMins[p]

        checkOut = ZDT6(netPars)

        # rescale for ANN
        for co in range(len(checkOut)):
            checkOut[co] = (checkOut[co] - self.smpMins[self.nPars + co]) / (
                        self.smpMaxs[self.nPars + co] - self.smpMins[self.nPars + co])

        # compute the error
        delta = 0
        delta += abs(netOuts[0] - checkOut[0])
        delta += abs(netOuts[1] - checkOut[1])

        self.writeToLog(f"Doing CFD check no. {self.toCompare.index(i)} with parameters {netPars}\n")
        self.writeToLog(f"no. {self.toCompare.index(i)} ANN outs were {netOuts}\n")
        self.writeToLog(f"no. {self.toCompare.index(i)} CFD outs were {checkOut} delta {delta}\n")
        self.writeToLog(f"CFD check no. {self.toCompare.index(i)} done\n")

        return delta

    def optimizeAndFindBestDNN(self, netStructs, netNms, netDirs, smpNondoms):
        # prepare
        lError = 1e3
        bestNet = ""

        # loop over net architectures
        for n in range(len(netStructs)):
            # load architecture, name and save directory
            netStruct = netStructs[n]
            netNm = netNms[n]
            netDir = netDirs[n]

            # run optimization
            parallelNum = self.nSeeds * self.nNN
            moea, nondoms = self.runDNNOptimization(netStruct, netNm, netDir, parallelNum)

            # convert nondominated solutions to array
            netNondoms = list()
            for i in range(len(nondoms)):
                netNondoms.append(nondoms[i].variables[:] + nondoms[i].objectives[:])

            # compare samples and dnn nondominated solutions
            dists = self.compareParetoFronts(netNondoms, smpNondoms)
            cError = sum(dists) / len(dists)

            # identify the best network
            if cError < lError:
                lError = cError
                bestNet = netNm

        return bestNet, netNondoms

    def runDNNOptimization(self, netStruct, netNm, netDir, parallelNum):
        # list net save directory
        ls = os.listdir(netDir)
        ls = [i for i in ls if ".png" not in i]

        # load all net seeds
        self.nets = list()
        for seed in ls:
            model = load_model(os.path.join(netDir, seed))
            self.nets.append(model)

        # construct optimization problem
        problem = plat.Problem(self.nPars, self.nObjs)
        problem.types[:] = [plat.Real(self.pMin, self.pMax)] * self.nPars
        problem.function = self.dnnEvaluation

        # run the optimization algorithm with archiving data
        with plat.MultiprocessingEvaluator(parallelNum) as evaluator:
            moea = plat.NSGAII(problem, population_size=self.popSize, offspring_size=self.offSize, evaluator=evaluator,
                               archive=plat.Archive())
            moea.run(self.nGens * self.popSize)

        # save data
        with open(netDir + "optimOut.plat", 'wb') as file:
            pickle.dump(
                [moea.population, moea.result, "NSGAII", problem],
                file,
                protocol=2
            )

        return moea, moea.result

    def runVerification(self, bestNet, stepDir):
        # choose random datapoints to verify
        self.toCompare = list()
        while len(self.toCompare) < self.nComps:
            toAdd = random.randint(0, self.popSize - 1)
            if toAdd not in self.toCompare:
                self.toCompare.append(toAdd)

        # load optimization data
        netDir = stepDir + bestNet + "/"
        with open(netDir + "optimOut.plat", 'rb') as file:
            [self.population, nondoms, algorithm, problem] = pickle.load(file, encoding="latin1")

        # run verification
        with multiprocessing.Pool(self.nComps) as p:
            deltas = p.map(self.smpEvaluation, self.toCompare)
        # deltas = []
        # for i in range(self.nComps):
        #     delta = self.smpEvaluation(self.toCompare[i])
        #     deltas.append(delta)

        # count non-evaluated cases
        bads = deltas.count(-1)
        deltas = [i for i in deltas if i >= 0]
        delta = sum(deltas)

        # choose substitute solutions for non-evaluated ones
        if bads > 0:
            # choose random datapoints
            secToCompare = list()
            while len(secToCompare) < bads:
                toAdd = random.randint(0, self.popSize - 1)
                if toAdd not in self.toCompare and toAdd not in secToCompare:
                    secToCompare.append(toAdd)

            self.toCompare = secToCompare[:]

            # run samples verification
            with multiprocessing.Pool(bads) as p:
                deltas = p.map(self.smpEvaluation, self.toCompare)
            # deltas = []
            # for i in range(bads):
            #     delta = self.smpEvaluation(self.toCompare[i])
            #     deltas.append(delta)

            # count still non-evaluated cases
            bads = deltas.count(-1)
            deltas = [i for i in deltas if i >= 0]
            delta += sum(deltas)

        return delta, bads

    def compareParetoFronts(self, netNondoms, smpNondoms):
        # prepare list to save
        dists = list()

        # loop over datapoints from samples nondominated solution
        for smpSol in smpNondoms:
            dist = 100
            # loop over datapoints from net nondominated solutions
            for netSol in netNondoms:
                potDist = np.linalg.norm(netSol[:self.nPars] - smpSol[:self.nPars])

                # find the nearest datapoint
                if potDist < dist:
                    dist = potDist

            # rescale with respect to parameter space size
            dists.append(dist / math.sqrt(self.nPars))

        return dists

    def findNondominatedSolutions(self, floatData, directions):
        # prepare problem
        problem = plat.Problem(self.nPars, self.nObjs)
        problem.types[:] = [plat.Real(self.pMin, self.pMax)] * self.nPars

        # convert array to population for platypus
        popData = list()
        for solution in floatData:
            individuum = plat.core.Solution(problem)
            individuum.variables = [solution[i] for i in range(self.nPars)]
            individuum.objectives = [solution[i] for i in range(self.nPars, len(solution))]
            individuum.evaluated = True
            popData.append(individuum)

        # let platypus find non-dominated solutions
        nondoms = plat.nondominated(popData)

        # convert population to array
        nonDomSolutions = list()
        for solution in nondoms:
            data = solution.variables[:] + solution.objectives[:]
            nonDomSolutions.append(data)
        nonDomSolutions = np.array(nonDomSolutions)

        return nonDomSolutions

    def prepareForNextIter(self, bestNet, prevSamTotal, nSamTotal):
        # lower variance
        self.rN -= self.dRN
        if self.rN < self.dRN:
            self.rN = self.dRN

        # get number of samples to add
        if (self.iteration - 1) >= len(self.deltaNSams):
            deltaNSam = self.deltaNSams[-1]
        else:
            deltaNSam = self.deltaNSams[self.iteration - 1]

        # save current number of samples and compute new
        prevSamTotal = nSamTotal
        self.nSam += deltaNSam
        nSamTotal = self.nSam / self.trainPro * 100

        # check if next iteration is last
        last = False
        if nSamTotal > self.maxSam:
            nSam = math.floor(self.maxSam * self.trainPro / 100)
            last = True

        # update mean number of neurons based on the best network found
        bestNs = bestNet.split("_")
        for i in range(self.nHidLay):
            self.nHid[i] = int(bestNs[i + 1])

        # update iteration counter
        self.iteration += 1

        return prevSamTotal, nSamTotal, last

    def loadAndScaleData(self, dataDir, dataNm, nPars, nObjs):
        """ function to load samples and scale them to be in <0,1> """

        # load samples
        with open(dataDir + dataNm, 'r') as file:
            data = file.readlines()

        # remove annotation row
        data = data[1::]

        # convert the data to numpy array
        dataNum = []
        for line in data:
            lineSpl = line.split(',')
            row = []
            for value in lineSpl:
                row.append(float(value))
            dataNum.append(row)

        dataNum = np.array(dataNum)

        # scale the data
        colMins = np.min(dataNum, axis=0)
        colMaxs = np.max(dataNum, axis=0)
        for rowInd in range(dataNum.shape[0]):
            for colInd in range(dataNum.shape[1]):
                dataNum[rowInd, colInd] = (dataNum[rowInd, colInd] - colMins[colInd]) / (
                            colMaxs[colInd] - colMins[colInd])

        # split and transpose
        source = dataNum[:, :nPars].T
        target = dataNum[:, nPars:nPars + nObjs].T

        return source, target

    def getScalesFromFile(self, dataDir, dataNm):
        """ function to get scales from the given file """

        # load samples
        with open(dataDir + dataNm, 'r') as file:
            data = file.readlines()

        # remove annotation row
        data = data[1::]

        # convert the data to numpy array
        dataNum = []
        for line in data:
            lineSpl = line.split(',')
            row = []
            for value in lineSpl:
                row.append(float(value))
            dataNum.append(row)

        dataNum = np.array(dataNum)

        # scale the data
        colMins = np.min(dataNum, axis=0)
        colMaxs = np.max(dataNum, axis=0)

        return colMins, colMaxs

    def prepOutDir(self, outDir, dirLstMk=[]):
        """ function to prepare the output directory """

        # prepare required directory if not already present
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        # prepare optional subdirectories if not already present
        for dr in dirLstMk:
            if not os.path.exists(outDir + dr):
                os.makedirs(outDir + dr)

    def plotRegressionGraph(self, netStructs, netNms, netDirs):
        # loop over required net directories
        for netDir in netDirs:
            # read directory
            ls = os.listdir(netDir)
            ls = [i for i in ls if not ".png" in i]

            # loop over net seeds
            for seed in ls:
                model = load_model(os.path.join(netDir, seed))
                out = model.predict(self.sourceTe.T).T

                # transpose data
                targetTe = self.targetTe.T
                out = out.T

                # plot the result ## NOTE: only prepared for two outputs
                mS = 7
                plt.plot(targetTe[:, 0], out[:, 0], 'o', ms=mS, color="tab:red")
                plt.plot(targetTe[:, 1], out[:, 1], '^', ms=mS, color="tab:green")
                plt.plot([-0.2, 1.2], [-0.2, 1.2], "k-")
                plt.xlabel("target data")
                plt.ylabel("estimated data")
                plt.title("Regression plot for NN")
                plt.legend(["f1", "f2"], loc="lower right")
                plt.xlim((-0.05, 1.05))
                plt.ylim((-0.05, 1.05))

                num = seed.split("_")[-1].split(".")[0]
                plt.savefig(netDir + "regressionPlot_" + num + ".png")
                plt.close()
