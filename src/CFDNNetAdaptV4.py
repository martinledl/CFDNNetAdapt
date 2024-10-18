import os
import sys
import math
import random
import datetime
from itertools import repeat
from sklearn.model_selection import train_test_split
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, InputLayer, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import multiprocessing

import platypusModV2 as plat
import levenberg_marquardt as lm


# ------------------ Known issues ------------------
# dnnEvaluation and smpEvaluation functions need to be defined in a different file and imported before passing them
# as a parameter.


class CFDNNetAdapt:
    def __init__(self, dnnEvaluation=None, smpEvaluation=None, nPars=2, nObjs=2, nOuts=2, mainDir="01_algoRuns/",
                 smpDir=r"00_prepData/", prbDir="ZDT6/", dataNm="10_platypusAllSolutions.dat", minMax="", nSam=2000,
                 deltaNSams=None, nNN=1, minN=8, maxN=32, nHidLay=3, tol=1e-5, iMax=3, dRN=1, nComps=16, nSeeds=1,
                 trainPro=70, valPro=20, testPro=10, kMax=10000, patience=100, validationFreq=1,
                 activationFunction="tanh", lm_optimizer=False, rEStop=1e-5, verbose=True, pMin=0.0, pMax=1.0,
                 offSize=100, popSize=100, nGens=250, drawTrainingPlot=False, toPlotReg=False, specRunDir=None,
                 saveTrainingHistory=False, doNotCreateRunDir=False, randomState=42, dropout=0.0, batchNorm=False,
                 netStructs=None, batch_size=512):

        # replace mutable default argument
        if deltaNSams is None:
            deltaNSams = [2000]

        # ------------------ user defined parameters ------------------

        # optimization problem
        self.nPars = nPars  # number of parameters
        self.nObjs = nObjs  # number of objectives
        self.nOuts = nOuts  # number of networks outputs

        # directories and data files
        self.mainDir = mainDir  # main save directory
        self.specRunDir = specRunDir  # specific run directory
        self.smpDir = smpDir  # directory with samples
        self.prbDir = prbDir  # specific data directory
        self.dataNm = dataNm  # name of the file with data
        self.minMax = minMax  # min-max scaling
        self.doNotCreateRunDir = doNotCreateRunDir  # whether to create run directory

        # CFDNNetAdapt hyperparameters
        self.nSam = nSam  # initial number of samples
        self.deltaNSams = deltaNSams  # factor to change number of samples, may be a list to change among iterations
        self.nNN = nNN  # number of neural networks to test
        self.minN = minN  # minimal number of neurons
        self.maxN = maxN  # maximal number of neurons
        self.nHidLay = nHidLay  # number of hidden layers
        self.tol = tol  # required tolerance
        self.iMax = iMax  # maximum number of iterations
        self.dRN = dRN  # factor to change variance of random number of neurons selection
        self.nComps = nComps  # number of verification checks
        self.nSeeds = nSeeds  # number of seeds
        self.patience = patience  # number of allowed validation fails before ending the training
        self.validationFreq = validationFreq  # frequency of validation checks (every n-th epoch)
        self.lm_optimizer = lm_optimizer  # whether to use Levenberg-Marquardt optimizer
        self.netStructs = netStructs  # list of neural network architectures (if supplied will not generate random ones)
        self.batch_size = batch_size

        # DNN parameters
        self.trainPro = trainPro  # percentage of samples used for training
        self.valPro = valPro  # percentage of samples used for validation
        self.testPro = testPro  # percentage of samples used for testing
        self.kMax = kMax  # maximum number of iterations for DNN training
        self.rEStop = rEStop  # required error for DNN validation
        self.batchNorm = batchNorm  # whether to use batch normalization
        self.dropout = dropout  # dropout rate
        self.activationFunction = activationFunction  # activation function for hidden layers
        self.dnnEvaluation = dnnEvaluation
        self.smpEvaluation = smpEvaluation

        # MOEA parameters
        self.pMin = pMin  # minimal allowed parameter value
        self.pMax = pMax  # maximal allowed parameter value
        self.offSize = offSize  # offspring size
        self.popSize = popSize  # population size
        self.nGens = nGens  # number of generations

        # options
        self.toPlotReg = toPlotReg  # whether to create regression plots, requires uncommenting matplotlib import
        self.randomState = randomState
        self.verbose = verbose  # print info about DNN training
        self.drawTrainingPlot = drawTrainingPlot  # draw training plot
        self.saveTrainingHistory = saveTrainingHistory  # save training history

        # ------------------ internal variables ------------------
        self._verbosityLevel = 1 if self.verbose else 0  # verbosity level for Keras
        self._nets = None  # list of neural networks

        self._netTransfers = [self.activationFunction] * self.nHidLay
        self._nHid = [(self.maxN + self.minN) / 2 for _ in range(self.nHidLay)]  # mean number of neurons for each layer
        self._rN = (self.maxN - self.minN) / 2  # variance for random number of neurons selection
        self._rN *= 0.5

        self._iteration = 1

        self._toCompare = None

        # ------------------ prepare directories ------------------
        self.prepOutDir(self.mainDir)
        if self.specRunDir is None:
            ls = os.listdir(self.mainDir)
            ls = sorted([i for i in ls if "run" in i])
            try:
                lastRunNum = int(ls[-1].split("_")[-1])
                self._runDir = self.mainDir + f"run_{(lastRunNum + 1):02d}/"
            except:
                self._runDir = self.mainDir + f"run_{(len(ls) + 1):02d}/"
        else:
            self._runDir = self.mainDir + self.specRunDir

        if not self.doNotCreateRunDir:
            self.prepOutDir(self._runDir)
            if self.verbose:
                print(f"Run directory created: {self._runDir}")

        # ------------------ prepare data ------------------
        # prepare mins and maxs for scaling
        self._smpMins, self._smpMaxs = self.getScalesFromFile(self.smpDir + self.prbDir, self.dataNm)

        # prepare samples
        self._source, self._target = self.loadAndScaleData(self.smpDir + self.prbDir, self.dataNm, self.nPars,
                                                           self.nOuts)
        self._souall, self._tarall = self.loadAndScaleData(self.smpDir + self.prbDir, self.dataNm, self.nPars,
                                                           self.nObjs)
        self._maxSam = np.shape(self._source)[1]  # maximum number of samples

        self._sourceTr = None
        self._targetTr = None
        self._sourceVl = None
        self._targetVl = None
        self._sourceTe = None
        self._targetTe = None

    def createNN(self, stepDir):
        self.writeToLog("Creating a new NN\n")
        newCheck = True
        skip = False
        netTry = 1

        # try to create new random architecture
        while newCheck:
            # compute allowed minimum and maximum
            nMins = list()
            nMaxs = list()
            for i in range(self.nHidLay):
                nMins.append(max(int(self._nHid[i] - self._rN), self.minN))
                nMaxs.append(min(int(self._nHid[i] + self._rN), self.maxN))

            # generate random number of neurons
            netStruct = [self.nPars]
            for i in range(self.nHidLay):
                netStruct += [random.randint(nMins[i], nMaxs[i])]
            netStruct += [self.nOuts]

            # create network name and save directory
            netNm = "_".join([str(i) for i in netStruct])
            netDir = stepDir + netNm + "/"

            # check for already existing networks
            if not os.path.exists(netDir):
                newCheck = False

            # ned if tried too many times
            elif netTry >= self.nNN:
                newCheck = False
                skip = True

            netTry += 1

        return netStruct, netNm, netDir, skip

    def createRandomDNNs(self, stepDir):
        netStructs = []
        netNms = []
        netDirs = []

        # create DNNs
        for _ in range(self.nNN):
            # create one architecture
            netStruct, netNm, netDir, skip = self.createNN(stepDir)

            # skip if duplicate
            if skip:
                continue

            # create network save directory
            self.prepOutDir(netDir)
            self.writeToLog("Created net " + str(netNm) + "\n")

            # save
            netStructs.append(netStruct)
            netNms.append(netNm)
            netDirs.append(netDir)

        return netStructs, netNms, netDirs

    def build_model(self, netStruct):
        model = Sequential()

        # First layer manages the input shape
        model.add(InputLayer(input_shape=(netStruct[0],)))

        # Add hidden layers
        for (neurons, activation) in zip(netStruct[1:-1], self._netTransfers):
            model.add(Dense(neurons, activation=activation))
            if self.dropout > 0:
                model.add(Dropout(self.dropout))

            if self.batchNorm:
                model.add(BatchNormalization())

        # Add output layer
        model.add(Dense(netStruct[-1], activation='linear'))

        if self.lm_optimizer:
            model_wrapper = lm.ModelWrapper(model)
            model_wrapper.compile(
                optimizer=SGD(learning_rate=1.0),
                loss=lm.MeanSquaredError())
            return model_wrapper
        else:
            model.compile(optimizer=Adam(), loss='mean_squared_error')
            return model

    def train_model(self, model, sourceTr, targetTr, sourceVl, targetVl):
        # Stops the training in case the validation loss is getting higher and revert to the best weights before this
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=self.patience,
                                       restore_best_weights=True)
        history = model.fit(sourceTr.T, targetTr.T, validation_data=(sourceVl.T, targetVl.T), epochs=self.kMax,
                            verbose=self._verbosityLevel, callbacks=[early_stopping], batch_size=self.batch_size,
                            validation_freq=self.validationFreq)
        return history

    def dnnSeedEvaluation(self, netStruct, seed):
        model = self.build_model(netStruct)
        self.writeToLog(f"Starting training DNN with seed {seed}\n")
        history = self.train_model(model, self._sourceTr, self._targetTr, self._sourceVl, self._targetVl)
        self.writeToLog(f"Training of DNN with seed {seed} finished\n")

        stepDir = self._runDir + f"step_{self._iteration:04d}/"
        netNm = "_".join([str(i) for i in netStruct])
        netDir = stepDir + netNm + "/"
        model.save(netDir + 'weights.keras')

        if self.drawTrainingPlot:
            self.plotTrainingGraph(history, netDir, self._iteration, seed)

        if self.saveTrainingHistory:
            with open(netDir + f"training_history-{seed:03d}.pkl", 'wb') as file:
                pickle.dump(
                    history.history,
                    file,
                    protocol=2
                )

        loss = model.evaluate(self._sourceVl.T, self._targetVl.T, verbose=self._verbosityLevel)
        self.writeToLog(f"Loss of DNN with seed {seed} is {loss}\n")

        return loss

    def handleGivenNetStructs(self, stepDir):
        netStructs = self.netStructs
        netNms = [f"{'_'.join([str(i) for i in netStruct])}" for netStruct in netStructs]
        netDirs = [stepDir + netNm + "/" for netNm in netNms]

        # create network save directory
        for i in range(len(netStructs)):
            self.prepOutDir(netDirs[i])
            self.writeToLog("Created net " + str(netNms[i]) + "\n")

        return netStructs, netNms, netDirs

    def run(self):
        self.startLog()
        if self.verbose:
            print(f"\nAvailable CPU cores: {self.get_available_cpu_cores()}\n")

        last = False
        epsilon = 1
        prevSamTotal = 0
        smpNondoms = None

        while epsilon > self.tol and self._iteration <= self.iMax:
            stepDir = self._runDir + f"step_{self._iteration:04d}/"
            self.prepOutDir(stepDir)
            # log
            self.writeToLog("Starting iteration " + str(self._iteration) + "\n")

            # compute number of samples used
            nSamTotal, trainLen, valLen, testLen = self.prepareSamples()

            # find pareto front from samples
            smpNondoms = self.getNondomSolutionsFromSamples(nSamTotal, smpNondoms)

            self.writeToLog(
                f"Using {self.nSam} training samples, {valLen} validation samples and {testLen} testSamples\n")

            # check the last best dnn
            if self._iteration > 1:
                self.writeToLog(f"Checking last best DNN from iteration {self._iteration - 1}\n")
                shouldExit = self.checkLastBestDNN(netNondoms, smpNondoms)
                if shouldExit:
                    return

            if self.netStructs is None:
                netStructs, netNms, netDirs = self.createRandomDNNs(stepDir)
            else:
                netStructs, netNms, netDirs = self.handleGivenNetStructs(stepDir)

            # CANNOT RUN IN PARALLEL!!! (does not work on Kraken)
            for n in range(self.nNN):
                for i in range(self.nSeeds):
                    cError = self.dnnSeedEvaluation(netStructs[n], i)

            if self.toPlotReg:
                self.plotRegressionGraph(netStructs, netNms, netDirs)

            self.writeToLog(f"Iteration {self._iteration} - Training finished\n")

            bestNet, netNondoms = self.optimizeAndFindBestDNN(netStructs, netNms, netDirs, smpNondoms)
            self.writeToLog(f"Iteration {self._iteration} - Best DNN found {bestNet}\n")

            delta, bads = self.runVerification(bestNet, stepDir)

            if bads == self.nComps:
                if (self._iteration - 1) >= len(self.deltaNSams):
                    deltaNSam = self.deltaNSams[-1]
                else:
                    deltaNSam = self.deltaNSams[self._iteration - 1]

                # self.nSam -= deltaNSam
                self._rN += self.dRN
            else:
                epsilon = delta / (self.nComps - bads)

            self.writeToLog(f"Last residual - {epsilon}\n\n")

            if last:
                self.writeToLog("Done. Maximum number of samples reached\n")
                self.finishLog()
                return

            prevSamTotal, nSamTotal, last = self.prepareForNextIter(bestNet, prevSamTotal, nSamTotal)

            # free memory used by keras
            tf.keras.backend.clear_session()

        self.writeToLog("Done. Required error reached\n")
        self.finishLog()

    @staticmethod
    def get_available_cpu_cores():
        return multiprocessing.cpu_count()

    def writeToLog(self, text):
        timestampString = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        with open(self._runDir + "log.out", 'a') as outFile:
            while len(text) > 1 and text[0] == "\n":
                outFile.write("\n")
                text = text[1:]

            outFile.write(f"[{timestampString}]: {text}")

    def startLog(self):
        self.writeToLog("startTime = " + str(datetime.datetime.now().strftime("%d/%m/%Y %X")) + "\n")
        self.writeToLog("===================SET UP=====================\n")

        self.writeToLog(f"\nAvailable CPU cores: {self.get_available_cpu_cores()}\n\n")

        # prepare things to write
        toWrite = [
            "nPars", "nOuts", "nObjs",
            "nSam", "deltaNSams",
            "nNN", "minN", "maxN", "nHidLay",
            "tol", "iMax", "dRN",
            "nComps", "nSeeds",
            "trainPro", "valPro", "testPro",
            "kMax", "rEStop", "patience",
            "pMin", "pMax",
            "offSize", "popSize", "nGens",
            "activationFunction", "_runDir",
            "batchNorm", "dropout"
        ]

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
        cSource = self._source[:, :nSamTotal]
        cTarget = self._target[:, :nSamTotal]

        # split data into training and remaining (validation + testing)
        self._sourceTr, source_rem, self._targetTr, target_rem = train_test_split(
            cSource.T, cTarget.T, train_size=self.trainPro / 100, random_state=self.randomState
        )

        # split remaining data into validation and testing
        if self.testPro > 0:
            self._sourceVl, self._sourceTe, self._targetVl, self._targetTe = train_test_split(
                source_rem, target_rem, test_size=self.testPro / (self.testPro + self.valPro),
                random_state=self.randomState
            )
            self._sourceTe = self._sourceTe.T
            self._targetTe = self._targetTe.T
        else:
            self._sourceVl = source_rem
            self._targetVl = target_rem
            self._sourceTe = np.empty((self._sourceTr.shape[0], 0))
            self._targetTe = np.empty((self._targetTr.shape[0], 0))

        # transpose back
        self._sourceTr = self._sourceTr.T
        self._targetTr = self._targetTr.T
        self._sourceVl = self._sourceVl.T
        self._targetVl = self._targetVl.T

        # get training, validation, and testing lengths
        trainLen = self._sourceTr.shape[1]
        valLen = self._sourceVl.shape[1]
        testLen = self._sourceTe.shape[1]

        return nSamTotal, trainLen, valLen, testLen

    def getNondomSolutionsFromSamples(self, nSamTotal, smpNondoms=None):
        # get samples added in this iteration
        aSource = self._souall[:, :nSamTotal]
        aTarget = self._tarall[:, :nSamTotal]

        # concatenate with last iteration nondominated solutions
        aAll = np.append(aSource, aTarget, axis=0)
        if self._iteration > 1:
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
            return True

        return False

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
            # parallelNum = self.nSeeds * self.nNN
            parallelNum = self.get_available_cpu_cores()
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
        ls = [i for i in ls if ".keras" in i]

        # load all net seeds
        self._nets = list()
        for seed in ls:
            model = load_model(os.path.join(netDir, seed))
            self._nets.append(model)

        # construct optimization problem
        problem = plat.Problem(self.nPars, self.nObjs)
        problem.types[:] = [plat.Real(self.pMin, self.pMax)] * self.nPars
        problem.function = self.dnnEvaluation
        problem.kwargs = {"nets": self._nets, "lm_optimizer": self.lm_optimizer, "smpMins": self._smpMins,
                          "smpMaxs": self._smpMaxs, "nObjs": self.nObjs, "nOuts": self.nOuts, "smpDir": self.smpDir,
                          "prbDir": self.prbDir, "minMax": self.minMax}

        # run the optimization algorithm with archiving data
        self.writeToLog(f"Starting NSGAII optimization with net {netNm}\n")
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
        self._toCompare = list()
        while len(self._toCompare) < self.nComps:
            toAdd = random.randint(0, self.popSize - 1)
            if toAdd not in self._toCompare:
                self._toCompare.append(toAdd)

        # load optimization data
        netDir = stepDir + bestNet + "/"
        with open(netDir + "optimOut.plat", 'rb') as file:
            [self.population, nondoms, algorithm, problem] = pickle.load(file, encoding="latin1")

        # run verification
        self.writeToLog(f"Starting verification with {bestNet}\n")
        parallelNum = self.get_available_cpu_cores()
        with multiprocessing.Pool(parallelNum) as p:
            deltas = p.map(self.smpEvaluation,
                           [(self.population, i, self._smpMins, self._smpMaxs, self.nPars) for i in self._toCompare])

        # count non-evaluated cases
        bads = deltas.count(-1)
        deltas = [i for i in deltas if i >= 0]
        delta = sum(deltas)

        self.writeToLog(f"Verification of {bestNet} finished, non-evaluated cases: {bads}\n")

        # choose substitute solutions for non-evaluated ones
        if bads > 0:
            # choose random datapoints
            secToCompare = list()
            while len(secToCompare) < bads:
                toAdd = random.randint(0, self.popSize - 1)
                if toAdd not in self._toCompare and toAdd not in secToCompare:
                    secToCompare.append(toAdd)

            self._toCompare = secToCompare[:]

            # run samples verification
            self.writeToLog(f"Starting verification with {bestNet} for substitute solutions\n")
            with multiprocessing.Pool(parallelNum) as p:
                deltas = p.map(self.smpEvaluation,
                               [(self.population, i, self._smpMaxs, self._smpMins, self.nPars) for i in
                                self._toCompare])

            # count still non-evaluated cases
            bads = deltas.count(-1)
            deltas = [i for i in deltas if i >= 0]
            delta += sum(deltas)

            self.writeToLog(
                f"Verification of {bestNet} for substitute solutions finished, still non-evaluated cases: {bads}\n")

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
        self._rN -= self.dRN
        if self._rN < self.dRN:
            self._rN = self.dRN

        # get number of samples to add
        if (self._iteration - 1) >= len(self.deltaNSams):
            deltaNSam = self.deltaNSams[-1]
        else:
            deltaNSam = self.deltaNSams[self._iteration - 1]

        # save current number of samples and compute new
        prevSamTotal = nSamTotal
        self.nSam += deltaNSam
        nSamTotal = self.nSam / self.trainPro * 100

        # check if next iteration is last
        last = False
        if nSamTotal > self._maxSam:
            nSam = math.floor(self._maxSam * self.trainPro / 100)
            last = True

        # update mean number of neurons based on the best network found
        bestNs = bestNet.split("_")
        for i in range(self.nHidLay):
            self._nHid[i] = int(bestNs[i + 1])

        # update iteration counter
        self._iteration += 1

        return prevSamTotal, nSamTotal, last

    @staticmethod
    def loadAndScaleData(dataDir, dataNm, nPars, nObjs):
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

    @staticmethod
    def getScalesFromFile(dataDir, dataNm):
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

    @staticmethod
    def prepOutDir(outDir, dirLstMk=None):
        """ function to prepare the output directory """

        # replace default mutable argument
        if dirLstMk is None:
            dirLstMk = []

        # prepare required directory if not already present
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        # prepare optional subdirectories if not already present
        for dr in dirLstMk:
            if not os.path.exists(outDir + dr):
                os.makedirs(outDir + dr)

    # NOTE: only prepared for two outputs
    def plotRegressionGraph(self, netDirs):
        # loop over required net directories
        for netDir in netDirs:
            # read directory
            ls = os.listdir(netDir)
            ls = [i for i in ls if ".keras" in i]

            # loop over net seeds
            for seed in ls:
                model = load_model(os.path.join(netDir, seed))

                if self.lm_optimizer:
                    model = lm.ModelWrapper(model)
                    model.compile(
                        optimizer=SGD(learning_rate=1.0),
                        loss=lm.MeanSquaredError())

                out = model.predict(self._sourceVl.T).T

                # transpose data
                targetVl = self._targetVl.T
                out = out.T

                # plot the result
                mS = 7
                plt.plot(targetVl[:, 0], out[:, 0], 'o', ms=mS, color="tab:red")
                plt.plot(targetVl[:, 1], out[:, 1], '^', ms=mS, color="tab:green")
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

    def plotTrainingGraph(self, history, outDir, iteration, seed):
        # plot training history
        train_history = history.history['loss']
        val_history = history.history['val_loss']
        time1 = np.arange(0, len(train_history))
        time2 = np.arange(0, len(val_history) * self.validationFreq, self.validationFreq)

        plt.plot(time1, train_history, label='train')
        plt.plot(time2, val_history, label='validation')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.title(f"Training history iteration {iteration}, seed {seed}")
        plt.savefig(outDir + f"trainingPlot_{seed:03d}.png")
        plt.close()

    def getRunDir(self):
        return self._runDir