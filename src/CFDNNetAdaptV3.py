
# import
import os
import sys
sys.path.insert(1, "../../thirdParty")
import math
import random
import datetime
import numpy as np
import dill as pickle
import multiprocessing
import pyrennModV3 as prn
import platypusModV2 as plat
from operator import itemgetter

#~ import matplotlib.pyplot as plt

# version 1 - more parallel evaluation
# version 2 - more separation in functions, renaming
# version 3 - adaptive samples step, archive moea data

class CFDNNetAdapt:
    def __init__(self):
        # optimization problem
        self.nPars = None # number of parameters
        self.nObjs = None # number of objectives
        self.nOuts = None # number of networks outputs

        # CFDNNetAdapt hyperparameters
        self.nSam = None # initial number of samples
        self.deltaNSams = None # factor to change number of samples
        self.nNN = None # number of neural networks to test
        self.minN = None # minimal number of neurons
        self.maxN = None # maximal number of neurons
        self.nHidLay = None # number of hidden layer
        self.tol = None # required tolerance
        self.iMax = None # maximum number of iterations
        self.dRN = None # factor to change rN
        self.nComps = None # number of samples verifications check
        self.nSeeds = None # number of seeds
        self.iteration = 1 # global iteration counter

        # DNN parameters
        self.trainPro = None # percentage of samples used for training
        self.valPro = None # percentage of samples used for validation
        self.testPro = None # percentage of samples used for testing
        self.kMax = None # maximum number of iterations for dnn training
        self.rEStop = None # required error for dnn validation
        self.dnnVerbose = False # print info about dnn training

        # MOEA parameters
        self.pMin = None # minimal allowed parameter value
        self.pMax = None # maximal allowed parameter value
        self.offSize = None # offspring size
        self.popSize = None # population size
        self.nGens = None # number of generations

        # directories and data files
        self.mainDir = None # main save directory
        self.smpDir = None # directory with samples
        self.prbDir = None # specific data directory
        self.dataNm = None # name of the file with data
        self.specRunDir = None # specified run directory, optional

        # evaluation functions
        self.dnnEvalFunc = None # custom function for dnn evaluation in optimization
        self.smpEvalFunc = None # custom function for sample evaluation in verification

        # flags
        self.toPlotReg = False # wheter to create regression plots

    def initialize(self):
        # prepare DNN specifics
        self.netTransfer = [prn.tanhTrans]*self.nHidLay
        self.nValFails = self.nHidLay*10
        self.nHid = [(self.maxN + self.minN)/2 for i in range(self.nHidLay)]
        self.rN = (self.maxN - self.minN)/2
        self.rN *= 0.5

        # prepare directories
        self.prepOutDir(self.mainDir)
        if self.specRunDir == None:
            ls = os.listdir(self.mainDir)
            ls = [i for i in ls if "run" in i]
            self.runDir = self.mainDir + "run_%02d/" % (len(ls)+1)
        else:
            self.runDir = self.mainDir + self.specRunDir
        self.prepOutDir(self.runDir)

        # prepare samples
        self.smpMins, self.smpMaxs = self.getScalesFromFile(self.smpDir + self.prbDir, self.dataNm)
        self.source, self.target = self.loadAndScaleData(self.smpDir + self.prbDir, self.dataNm, self.nPars, self.nOuts)
        self.souall, self.tarall = self.loadAndScaleData(self.smpDir + self.prbDir, self.dataNm, self.nPars, self.nObjs)
        self.maxSam = np.shape(self.source)[1]

    def run(self):
        # start log
        self.startLog()

        # run algorithm
        last = False
        epsilon = 1
        prevSamTotal = 0
        smpNondoms = None
        while epsilon > self.tol:
            # prepare step-directory to save data
            stepDir = self.runDir + "step_%04d/" % self.iteration
            self.prepOutDir(stepDir)
        
            # log
            self.outFile.write("Starting iteration " + str(self.iteration) + "\n")
        
            # compute number of samples used
            nSamTotal, trainLen, valLen, testLen = self.prepareSamples()

            # find pareto front from samples
            smpNondoms = self.getNondomSolutionsFromSamples(prevSamTotal, nSamTotal, smpNondoms)

            # log
            self.outFile.write("Using " + str(self.nSam) + " training samples, " + str(valLen) + " validation samples and " + str(testLen) + " testSamples\n")
            self.outFile.flush()

            # check the last best dnn
            if self.iteration > 1:
                self.checkLastBestDNN(netNondoms, smpNondoms)

            # create random DNNs
            netStructs, netNms, netDirs = self.createRandomDNNs(stepDir)
    
            # prepare arguments for training DNNs
            arguments = self.packDataForDNNTraining(netStructs)

            # train DNNs
            parallelNum = self.nSeeds*len(netStructs)
            with multiprocessing.Pool(parallelNum) as p:
                cErrors = p.map(self.dnnSeedEvaluation, arguments)

            # plot regression if required
            if self.toPlotReg:
                self.plotRegressionGraph(netStructs, netNms, netDirs)
    
            self.outFile.write("Iteration " + str(self.iteration) + " - Training finished \n")
            self.outFile.flush()
    
            # run optimizations and find the best DNN
            bestNet, netNondoms = self.optimizeAndFindBestDNN(netStructs, netNms, netDirs, smpNondoms)

            # log
            self.outFile.write("Iteration " + str(self.iteration) + " - Best DNN found "  + bestNet + "\n")
            self.outFile.flush()
    
            # verify DNN result
            delta, bads = self.runVerification(bestNet, stepDir)

            # if all cases non-evaluated -- restart step
            if bads == self.nComps:
                if (self.iteration-1) >= len(self.deltaNSams):
                    deltaNSam = self.deltaNSams[-1]
                else:
                    deltaNSam = self.deltaNSams[self.iteration-1]

                self.nSam -= deltaNSam
                self.rN += self.dRN
    
            else:
                epsilon = delta/(self.nComps - bads)
    
            # log
            self.outFile.write("Last residual - " + str(epsilon) + "\n\n")
            self.outFile.flush()

            # check second termination condition
            if last:
                self.outFile.write("Done. Maximum number of samples reached\n")
                self.finishLog()
                exit()
    
            # update parameters
            prevSamTotal, nSamTotal, last = self.prepareForNextIter(bestNet, prevSamTotal, nSamTotal)

        self.outFile.write("Done. Required error reached\n")
        self.finishLog()

    def startLog(self):
        self.outFile = open(self.runDir + "log.out", 'w')
        self.outFile.write("\nstartTime = " + str(datetime.datetime.now().strftime("%d/%m/%Y %X")) + "\n")
        self.outFile.write("===================SET UP=====================\n")
        toWrite = [
                "nPars", "nOuts", "nObjs",
                "nSam","deltaNSams",
                "nNN","minN","maxN","nHidLay",
                "tol","iMax","dRN",
                "nComps","nSeeds",
                "trainPro","valPro","testPro",
                "kMax","rEStop","nValFails",
                "pMin","pMax",
                "offSize","popSize","nGens"]

        for thing in toWrite:
            self.outFile.write(thing + " = " + str(eval("self." + thing)) + "\n")
        self.outFile.write("\n")
        self.outFile.flush()

    def finishLog(self):
        self.outFile.write("==============================================\n")
        self.outFile.write("endTime = " + str(datetime.datetime.now().strftime("%d/%m/%Y %X")) + "\n")
        self.outFile.close()

    def prepareSamples(self):
        nSamTotal = int(self.nSam/self.trainPro*100)
        
        # prepare samples
        cSource = self.source[:,:nSamTotal]
        cTarget = self.target[:,:nSamTotal]
        
        trainLen = int(self.trainPro/100*nSamTotal)
        valLen = int(self.valPro/100*nSamTotal)
        testLen = nSamTotal - trainLen - valLen
        
        self.sourceTr = cSource[:,:trainLen]
        self.targetTr = cTarget[:,:trainLen]
        
        self.sourceVl = cSource[:,trainLen:trainLen+valLen]
        self.targetVl = cTarget[:,trainLen:trainLen+valLen]
        
        self.sourceTe = cSource[:,trainLen+valLen:]
        self.targetTe = cTarget[:,trainLen+valLen:]

        return nSamTotal, trainLen, valLen, testLen

    def getNondomSolutionsFromSamples(self, prevSamTotal, nSamTotal, smpNondoms = None):
        aSource = self.souall[:,prevSamTotal:nSamTotal]
        aTarget = self.tarall[:,prevSamTotal:nSamTotal]
        aAll = np.append(aSource, aTarget, axis = 0)
        if self.iteration > 1:
            aAll = np.concatenate((smpNondoms.T,aAll), axis = 1)

        nondoms = self.findNondominatedSolutions(aAll.T, [1,1])
        return nondoms

    def checkLastBestDNN(self, netNondoms, smpNondoms):
        dists = self.compareParetoFronts(netNondoms, smpNondoms)
        
        pError = sum(dists)/len(dists)
        self.outFile.write("Error of best DNN from last iteration is " + str(pError) + "\n")
        self.outFile.flush()
        
        if pError < self.tol:
            self.outFile.write("Done. Last best DNN error < " + str(self.tol) + "\n")
            self.finishLog()
            exit()

    def createRandomDNNs(self, stepDir):
        netStructs = list()
        netNms = list()
        netDirs = list()
        for n in range(self.nNN):
            netStruct, netNm, netDir, skip = self.createNN(stepDir)
        
            if skip:
                continue

            # create network save directory
            self.prepOutDir(netDir)
            self.outFile.write("Created net " + str(netNm) + "\n")

            # save
            netStructs.append(netStruct)
            netNms.append(netNm)
            netDirs.append(netDir)

        return netStructs, netNms, netDirs

    def createNN(self, stepDir):
        newCheck = True
        skip = False
        netTry = 1
        while newCheck:
            # generate the network architecture
            nMins = list()
            nMaxs = list()
            for i in range(self.nHidLay):
                nMins.append(max(int(self.nHid[i] - self.rN), self.minN))
                nMaxs.append(min(int(self.nHid[i] + self.rN), self.maxN))
        
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
        
            elif netTry >= self.nNN:
                newCheck = False
                skip = True
        
            netTry += 1

        return netStruct, netNm, netDir, skip

    def packDataForDNNTraining(self, netStructs):
        arguments = list()
        for n in range(len(netStructs)):
            for i in range(self.nSeeds):
                argument = ([netStructs[n],
                    self.netTransfer,
                    self.sourceTr, self.targetTr,
                    self.sourceVl, self.targetVl,
                    self.sourceTe, self.targetTe,
                    self.kMax, self.rEStop, self.nValFails, self.dnnVerbose,
                    self.runDir, self.iteration,
                    i])
                arguments.append(argument)

        return arguments

    @staticmethod
    def dnnSeedEvaluation(args):
        """ function to evaluate dnn seed """
    
        # unpack agruments
        netStruct, netTransfer, sourceTr, targetTr, sourceVl, targetVl, sourceTe, targetTe, kMax, rEStop, nValFails, dnnVerbose, runDir, iteration, seed = args 

        # create the network
        net = prn.CreateNN(netStruct, transfers = netTransfer)
    
        # train the network
        prn.train_LMWithValData(
            sourceTr, targetTr,
            sourceVl, targetVl,
            net,
            verbose = dnnVerbose, k_max = kMax, RelE_stop = rEStop, maxValFails = nValFails
        )
    
        # save the network
        stepDir = runDir + "step_%04d/" % iteration
        netNm = "_".join([str(i) for i in netStruct])
        netDir = stepDir + netNm + "/"
        with open(netDir + '%s_%03d.dnn'%(netNm, seed), 'wb') as file:
            pickle.dump(
                [net],
                file,
                protocol=2
            )
    
        # test the DNN on testing data
        nOuts, testLen = np.shape(targetTe)
        out = np.array(prn.NNOut(sourceTe, net))
        if np.ndim(out) == 1:
            out = np.expand_dims(out, axis = 1)
            out = out.T
    
        # compute the seed and total error
        cError = 0
        for i in range(testLen):
            for j in range(nOuts):
                cError += abs(out[j,i] - targetTe[j,i])
        cError /= (nOuts*testLen)
    
        return cError

    def optimizeAndFindBestDNN(self, netStructs, netNms, netDirs, smpNondoms):
        lError = 1e3
        bestNet = ""
        for n in range(len(netStructs)):
            netStruct = netStructs[n]
            netNm = netNms[n]
            netDir = netDirs[n]

            # run optimization
            parallelNum = self.nSeeds*self.nNN
            moea, nondoms = self.runDNNOptimization(netStruct, netNm, netDir, parallelNum)

            # find nondominated solutions 
            netNondoms = list()
            for i in range(len(nondoms)):
                netNondoms.append(nondoms[i].variables[:] + nondoms[i].objectives[:])
    
            # compare samples and dnn nondominated solutions
            dists = self.compareParetoFronts(netNondoms, smpNondoms)
            cError = sum(dists)/len(dists)

            # identify the best network
            if cError < lError:
                lError = cError
                bestNet = netNm

        return bestNet, netNondoms
    
    def runDNNOptimization(self, netStruct, netNm, netDir, parallelNum):
        # prepare for optimization
        ls = os.listdir(netDir)
        ls = [i for i in ls if not ".png" in i]
    
        # load all net seeds
        self.nets = list()
        for seed in ls:
            with open(netDir + seed, 'rb') as file:
                [net] = pickle.load(file)
    
            self.nets.append(net)
    
        # construct optimization problem
        problem = plat.Problem(self.nPars, self.nObjs)
        problem.types[:] = [plat.Real(self.pMin,self.pMax)]*self.nPars
        problem.function = self.dnnEvalFunc

        # run the optimization algorithm
        with plat.MultiprocessingEvaluator(parallelNum) as evaluator:
            moea = plat.NSGAII(problem, population_size = self.popSize, offspring_size = self.offSize, evaluator = evaluator, archive = plat.Archive())
            moea.run(self.nGens*self.popSize)

        # save data
        with open(netDir + "optimOut.plat", 'wb') as file:
            pickle.dump(
                [moea.population, moea.result, "NSGAII", problem],
                file,
                protocol=2
            )

        return moea, moea.result

    def runVerification(self, bestNet, stepDir):
        self.toCompare = list()
        while len(self.toCompare) < self.nComps:
            toAdd = random.randint(0, self.popSize-1)
            if not toAdd in self.toCompare:
                self.toCompare.append(toAdd)

        # load optimization data
        netDir = stepDir + bestNet + "/"
        with open(netDir + "optimOut.plat", 'rb') as file:
            [self.population, nondoms, algorithm, problem] = pickle.load(file, encoding="latin1")
    
        # run samples verification
        with multiprocessing.Pool(self.nComps) as p:
            deltas = p.map(self.smpEvalFunc, self.toCompare)
    
        # count non-evaluated cases
        bads = deltas.count(-1)
        deltas = [i for i in deltas if i >= 0]
        delta = sum(deltas)
    
        # choose substitute solutions for verification
        if bads > 0:
            secToCompare = list()
            while len(secToCompare) < bads:
                toAdd = random.randint(0, self.popSize-1)
                if not toAdd in self.toCompare and not toAdd in secToCompare:
                    secToCompare.append(toAdd)
    
            self.toCompare = secToCompare[:]
    
            # run samples verification
            with multiprocessing.Pool(bads) as p:
                deltas = p.map(self.smpEvalFunc, self.toCompare)
    
            # count non-evaluated cases
            bads = deltas.count(-1)
            deltas = [i for i in deltas if i >= 0]
            delta += sum(deltas)
    
        return delta, bads

    def compareParetoFronts(self, netNondoms, smpNondoms):
        dists = list()
        for smpSol in smpNondoms:
            dist = 100
            for netSol in netNondoms:
                potDist = np.linalg.norm(netSol[:self.nPars] - smpSol[:self.nPars])
                if potDist < dist:
                    dist = potDist
        
            dists.append(dist/math.sqrt(self.nPars))

        return dists

    def findNondominatedSolutions(self, floatData, directions):
        nonDomSolutions = list()
        checked = np.zeros((len(floatData),))

        for p in range(len(floatData)):
            point = floatData[p]
            if p > 0:
                dist = np.linalg.norm(point - floatData[p-1])
                if dist < 0.1*self.tol:
                    continue

            dominated = False
            for c in range(len(floatData)):
                if checked[len(floatData)-1-c] == 1: # dominated solution
                    continue

                cPoint = np.flipud(floatData)[c]
                aux = 0
                for i in range(self.nObjs):
                    if directions[i] > 0:
                        if point[i+self.nPars] > cPoint[i+self.nPars]:
                            aux += 1
                    elif directions[i] < 0:
                        if point[i+self.nPars] < cPoint[i+self.nPars]:
                            aux += 1

                if aux == self.nObjs:
                    checked[p] = 1
                    dominated = True
                    break

            if not dominated:
                nonDomSolutions.append(point)

        nonDomSolutions = sorted(nonDomSolutions, key = itemgetter(self.nPars + 0))
        nonDomSolutions = np.array(nonDomSolutions)

        return nonDomSolutions

    def prepareForNextIter(self, bestNet, prevSamTotal, nSamTotal):
        self.rN -= self.dRN
        if self.rN < self.dRN:
            self.rN = self.dRN
    
        if (self.iteration-1) >= len(self.deltaNSams):
            deltaNSam = self.deltaNSams[-1]
        else:
            deltaNSam = self.deltaNSams[self.iteration-1]

        prevSamTotal = nSamTotal
        self.nSam += deltaNSam
        nSamTotal = self.nSam/self.trainPro*100

        last = False
        if nSamTotal > self.maxSam:
            nSam = math.floor(self.maxSam*self.trainPro/100)
            last = True
    
        bestNs = bestNet.split("_")
        for i in range(self.nHidLay):
            self.nHid[i] = int(bestNs[i+1])
    
        self.iteration += 1

        return prevSamTotal, nSamTotal, last

    def loadAndScaleData(self, dataDir, dataNm, nPars, nObjs):
        """ function to load samples and scale then in <0,1> """
    
        # load samples
        with open(dataDir + dataNm,'r') as file:
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
                dataNum[rowInd, colInd] = (dataNum[rowInd, colInd]-colMins[colInd])/(colMaxs[colInd]-colMins[colInd])
    
        source = dataNum[:, :nPars].T
        target = dataNum[:, nPars:nPars+nObjs].T

        return source,target

    def getScalesFromFile(self, dataDir, dataNm):
        """ function to get scales from the given file """
    
        # load samples
        with open(dataDir + dataNm,'r') as file:
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
        colMins = np.min(dataNum, axis = 0)
        colMaxs = np.max(dataNum, axis = 0)
    
        return colMins, colMaxs

    def prepOutDir(self, outDir, dirLstMk = []):
        """ function to prepare the output directory """
    
        if not os.path.exists(outDir): # check if outDir exists
            os.makedirs(outDir)
    
        for dr in dirLstMk:
            if not os.path.exists(outDir + dr): # if it does not already exist
                os.makedirs(outDir + dr)

    def plotRegressionGraph(self, netStructs, netNms, netDirs):
        for netDir in netDirs:
            # read directory
            ls = os.listdir(netDir)
            ls = [i for i in ls if not ".png" in i]

            # load and plot all net seeds
            for seed in ls:
                with open(netDir + seed, 'rb') as file:
                    [net] = pickle.load(file)

                # test the network on validation data
                out = np.array(prn.NNOut(self.sourceTe,net))

                # transpose
                targetTe = self.targetTe.T
                out = out.T

                # plot the result ## NOTE: only prepared for two outputs
                mS = 7
                plt.plot(targetTe[:,0], out[:,0], 'o', ms = mS, color = "tab:red")
                plt.plot(targetTe[:,1], out[:,1], '^', ms = mS, color = "tab:green")
                plt.plot([-0.2, 1.2], [-0.2, 1.2], "k-")
                plt.xlabel("target data")
                plt.ylabel("estimated data")
                plt.title("Regression plot for NN")
                plt.legend(["f1", "f2"], loc = "lower right")
                plt.xlim((-0.05, 1.05))
                plt.ylim((-0.05, 1.05))

                num = seed.split("_")[-1].split(".")[0]
                plt.savefig(netDir + "regressionPlot_" + num + ".png")
                plt.close()
