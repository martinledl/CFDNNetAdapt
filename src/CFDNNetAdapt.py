
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
import platypusModV3 as plat
from operator import itemgetter

class CFDNNetAdapt:
    def __init__(self):
        # optimization problem
        self.nPars = None # number of parameters
        self.nObjs = None # number of objectives
        self.nOuts = None # number of networks outputs

        # CFDNNetAdapt hyperparameters
        self.nSam = None # initial number of samples
        self.deltaNSam = None # factor to change number of samples
        self.nNN = None # number of neural networks to test
        self.minN = None # minimal number of neurons
        self.maxN = None # maximal number of neurons
        self.nHidLay = None # number of hidden layer
        self.tol = None # required tolerance
        self.iMax = None # maximum number of iterations
        self.dRN = None # factor to change rN
        self.nComps = None # number of CFD verifications check
        self.nSeeds = None # number of seeds
        self.iteration = 1 # global iteration counter

        # DNN parameters
        self.trainPro = None # percentage of samples used for training
        self.valPro = None # percentage of samples used for validation
        self.testPro = None # percentage of samples used for testing
        self.kMax = None # maximum number of iterations for dnn training
        self.rEStop = None # required error for dnn validation

        # MOEA parameters
        self.pMin = None # minimal allowed parameter value
        self.pMax = None # maximal allowed parameter value
        self.offSize = None # offspring size
        self.popSize = None # population size
        self.nGens = None # number of generations

        # directories and data files
        self.mainDir = None # main save directory
        self.cfdDir = None # directory with CFD data
        self.prbDir = None # specific CFD data directory
        self.dataNm = None # name of the file with CFD data

        # evaluation functions
        self.dnnEvalFunc = None # custom function for dnn evaluation in optimization
        self.cfdEvalFunc = None # custom function for cfd evaluation in verification

    def initialize(self):
        # prepare DNN specifics
        self.netTransfer = [prn.tanhTrans]*self.nHidLay
        self.nValFails = self.nHidLay*10
        self.nHid = [(self.maxN + self.minN)/2 for i in range(self.nHidLay)]
        self.rN = (self.maxN - self.minN)/2
        self.rN *= 0.5

        # prepare directories
        self.prepOutDir(self.mainDir)
        ls = os.listdir(self.mainDir)
        ls = [i for i in ls if "run" in i]
        self.runDir = self.mainDir + "run_%02d/" % (len(ls)+1)
        self.prepOutDir(self.runDir)

        # prepare CFD samples
        self.cfdMins, self.cfdMaxs = self.getScalesFromFile(self.cfdDir + self.prbDir, self.dataNm)
        self.source, self.target = self.loadAndScaleData(self.cfdDir + self.prbDir, self.dataNm, self.nPars, self.nOuts)
        self.souall, self.tarall = self.loadAndScaleData(self.cfdDir + self.prbDir, self.dataNm, self.nPars, self.nObjs)
        self.maxSam = np.shape(self.source)[1]

    def run(self):
        # start log
        self.startLog()

        # run algorithm
        last = False
        epsilon = 1
        while epsilon > self.tol:
            # prepare step-directory to save data
            stepDir = self.runDir + "step_%04d/" % self.iteration
            self.prepOutDir(stepDir)
        
            # log
            self.outFile.write("Starting iteration " + str(self.iteration) + "\n")
        
            # compute number of samples used
            nSamTotal = int(self.nSam/self.trainPro*100)
        
            # prepare CFD samples
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

            # find CFD-based pareto front
            aSource = self.souall[:,:nSamTotal]
            aTarget = self.tarall[:,:nSamTotal]
            aAll = np.append(aSource, aTarget, axis = 0)
            nondoms = self.findNonDominatedSolutions(aAll.T, [1,1])
        
            # log
            self.outFile.write("Using " + str(self.nSam) + " training samples, " + str(valLen) + " validation samples and " + str(testLen) + " testSamples\n")
            self.outFile.flush()

            # check the last best dnn
            if self.iteration > 1:
                dists = list()
                for cfdsol in nondoms:
                    dist = 100
                    for netsol in netnondoms:
                        potDist = np.linalg.norm(netsol[:self.nPars] - cfdsol[:self.nPars])
                        if potDist < dist:
                            dist = potDist
        
                    dists.append(dist/math.sqrt(self.nPars))
        
                pError = sum(dists)/len(dists)
                self.outFile.write("Error of best DNN from last iteration is " + str(pError) + "\n")
                self.outFile.flush()
        
                if pError < self.tol:
                    self.outFile.write("Done. Last best DNN error < " + str(self.tol) + "\n")
                    self.finishLog()
                    exit()

            # create random DNNs and find the best
            lError = 1e3
            bestNet = ""
        
            for n in range(self.nNN):
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
        
                if skip:
                    continue

                # create network save directory
                self.prepOutDir(netDir)
                self.outFile.write("Working with net " + str(netNm) + "\n")
    
                # train dnns
                argument = [([netStruct,
                    self.netTransfer,
                    self.sourceTr, self.targetTr,
                    self.sourceVl, self.targetVl,
                    self.sourceTe, self.targetTe,
                    self.kMax, self.rEStop, self.nValFails,
                    self.runDir, self.iteration,
                    i]) for i in range(self.nSeeds)]
                with multiprocessing.Pool(self.nSeeds) as p:
                    cErrors = p.map(self.dnnSeedEvaluation, argument)
    
                self.outFile.write("Iteration " + str(self.iteration) + " - Training finished \n")
                self.outFile.flush()
    
                # prepare for optimization
                ls = os.listdir(netDir)
                ls = [i for i in ls if not ".png" in i]
    
                self.nets = list()
                for seed in ls:
                    with open(netDir + seed, 'rb') as file:
                        [net] = pickle.load(file)
    
                    self.nets.append(net)
    
                # optimization problem construction
                problem = plat.Problem(self.nPars, self.nObjs)
                problem.types[:] = [plat.Real(self.pMin,self.pMax)]*self.nPars
                problem.function = self.dnnEvalFunc
    
                # run the optimization algorithm
                with plat.MultiprocessingEvaluator(self.nComps) as evaluator:
                    moea = plat.NSGAII(problem, population_size = self.popSize, offspring_size = self.offSize, evaluator = evaluator)
                    moea.run(self.nGens*self.popSize)

                # save data
                with open(netDir + "optimOut.plat", 'wb') as file:
                    pickle.dump(
                        [moea.population, "NSGAII", problem],
                        file,
                        protocol=2
                    )
    
                # find nondominated solutions 
                self.result = moea.population
                netOuts = list()
                for i in range(len(self.result)):
                    netOuts.append(self.result[i].variables[:] + self.result[i].objectives[:])
                netOuts = np.array(netOuts)
                netnondoms = self.findNonDominatedSolutions(netOuts, [1,1])
    
                # compare CFD-based and dnn nondominated solutions
                dists = list()
                for cfdsol in nondoms:
                    dist = 100
                    for netsol in netnondoms:
                        potDist = np.linalg.norm(netsol-cfdsol)
                        if potDist < dist:
                            dist = potDist
    
                    dists.append(dist/math.sqrt(self.nPars+self.nObjs))
                cError = sum(dists)/len(dists)
    
                # identify the best network
                if cError < lError:
                    lError = cError
                    bestNet = netNm
    
            # log
            self.outFile.write("Iteration " + str(self.iteration) + " - Best DNN found "  + bestNet + "\n")
            self.outFile.flush()
    
            # choose soutions for verification
            self.toCompare = list()
            while len(self.toCompare) < self.nComps:
                toAdd = random.randint(0, self.popSize-1)
                if not toAdd in self.toCompare:
                    self.toCompare.append(toAdd)

            # load optimization data
            netDir = stepDir + bestNet + "/"
            with open(netDir + "optimOut.plat", 'rb') as file:
                [self.result, algorithm, problem] = pickle.load(file, encoding="latin1")
    
            # run CFD-based verification
            with multiprocessing.Pool(self.nComps) as p:
                deltas = p.map(self.cfdEvalFunc, self.toCompare)
    
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
    
                # run CFD-based verification
                with multiprocessing.Pool(bads) as p:
                    deltas = p.map(self.cfdEvalFunc, self.toCompare)
    
                # count non-evaluated cases
                bads = deltas.count(-1)
                deltas = [i for i in deltas if i >= 0]
                delta += sum(deltas)
    
            # if all cases non-evaluated -- restart step
            if bads == self.nComps:
                self.nSam -= self.deltaNSam
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
            self.rN -= self.dRN
            if self.rN < self.dRN:
                self.rN = self.dRN
    
            self.nSam += self.deltaNSam
            nSamTotal = self.nSam/self.trainPro*100
            if nSamTotal > self.maxSam:
                nSam = math.floor(self.maxSam*self.trainPro/100)
                last = True
    
            bestNs = bestNet.split("_")
            for i in range(self.nHidLay):
                self.nHid[i] = int(bestNs[i+1])
    
            self.iteration += 1

        self.outFile.write("Done. Required error reached\n")
        self.finishLog()

    def startLog(self):
        self.outFile = open(self.runDir + "log.out", 'w')
        self.outFile.write("\nstartTime = " + str(datetime.datetime.now().strftime("%d/%m/%Y %X")) + "\n")
        self.outFile.write("===================SET UP=====================\n")
        toWrite = [
                "nPars", "nOuts", "nObjs",
                "nSam","deltaNSam",
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

    @staticmethod
    def dnnSeedEvaluation(args):
        """ function to evaluate dnn seed """
    
        # unpack agruments
        netStruct, netTransfer, sourceTr, targetTr, sourceVl, targetVl, sourceTe, targetTe, kMax, rEStop, nValFails, runDir, iteration, seed = args 

        # create the network
        net = prn.CreateNN(netStruct, transfers = netTransfer)
    
        # train the network
        prn.train_LMWithValData(
            sourceTr, targetTr,
            sourceVl, targetVl,
            net,
            verbose = False, k_max = kMax, RelE_stop = rEStop, maxValFails = nValFails
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

    def findNonDominatedSolutions(self, floatData, directions):
        nonDomSolutions = list()

        for point in floatData:
            dominated = False
            for cPoint in floatData:
                aux = 0
                for i in range(self.nObjs):
                    if directions[i] > 0:
                        if point[i+self.nPars] > cPoint[i+self.nPars]:
                            aux += 1
                    elif directions[i] < 0:
                        if point[i+self.nPars] < cPoint[i+self.nPars]:
                            aux += 1

                if aux == self.nObjs:
                    dominated = True

            if not dominated:
                nonDomSolutions.append(point)

        nonDomSolutions = sorted(nonDomSolutions, key = itemgetter(self.nPars + 0))
        nonDomSolutions = np.array(nonDomSolutions)

        return nonDomSolutions

    def loadAndScaleData(self, dataDir, dataNm, nPars, nObjs):
        """ function to load CFD samples and scale then in <0,1> """
    
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
