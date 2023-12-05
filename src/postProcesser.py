
import os
import sys
import csv
import math
import glob
import subprocess
import numpy as np
import shutil as sh
from operator import itemgetter

# auxiliary functions for post processing
class postProcesser:
    def __init__(self, caseDir, logDir = "./logs/", devnull = True):
        # caseDir - path to the case directory
        self.caseDir = caseDir
        self.logDir = logDir
        self.devnull = devnull

        # create a folder to save not-so-important logs there
        if not os.path.exists(logDir) and not self.devnull:
            os.makedirs(logDir)

        # get the version of python running this script
        self.pyVersion = sys.version[0]

        # get the local openfoam version
        ls = os.listdir("/opt")
        potVersions = list()

        for folder in ls:
            if os.path.isdir("/opt/"+folder):
                if "openfoam" in folder.lower() and not "paraview" in folder.lower():
                    potVersions.append(folder)

        foamToLoad = None
        for foam in potVersions: # prefere version 8
            if "8" in foam:
                check = os.listdir("/opt/"+foam+"/etc/") # check if there is something to load
                if "bashrc" in check:
                    foamToLoad = foam

            elif "6" in foam:
                check = os.listdir("/opt/"+foam+"/etc/") # check if there is something to load
                if "bashrc" in check:
                    foamToLoad = foam

            elif "4" in foam and not foamToLoad:
                check = os.listdir("/opt/"+foam+"/etc/") # check if there is something to load
                if "bashrc" in check:
                    foamToLoad = foam

        self.foamToLoad = foamToLoad

    # auxiliar functions
    def makeMeansOfDuplicatedXs(self, lst, vicaVersa = False):
        # lst - list of two lists, [[xs], [ys]]

        if vicaVersa:
            xLst = lst[1]
            yLst = lst[0]

        else:
            xLst = lst[0]
            yLst = lst[1]

        xM = []
        yM = []

        sorSet = sorted(set(xLst))

        for x in sorSet:

            yTemp = []

            for id, xD in enumerate(xLst):
                if xD == x:
                    yTemp.append(yLst[id])

            xM.append(x)
            yM.append(np.mean(yTemp))

        if vicaVersa:
            lstM = [yM, xM]

        else:
            lstM = [xM, yM]

        return lstM

    # checks
    def meshCheck(self, blockMesh = True, checkMesh = True):
        badMesh = list()
        
        if blockMesh:
            badMesh.append(self.findInLog("log.blockMesh", "Failed"))

        if checkMesh:
            badMesh.append(self.findInLog("log.checkMesh", "Failed"))

        if True in badMesh:
            return False

        else:
            return True

    def convergenceCheck(self):
        convergence = self.findInLog("log.simpleFoam", "solution converged")

        if convergence:
            return True

        else:
            return False

    def finalResidualCheck(self, field, threshold):
        fileName = self.caseDir + "postProcessing/residuals/0/residuals.dat"
        with open(fileName, 'r') as file:
            data = file.readlines()

        cols = data[1][2:].split()
        finalRes = data[-1].split('\t')

        del data

        value = float(finalRes[cols.index(field)])
        boolean = value < threshold

        return boolean

    def isComputed(self):
        fileName = self.caseDir + "system/controlDict"
        with open(fileName, 'r') as file:
            data = file.readlines()

        for line in data:
            if "application" in line:
                application = line.split()[-1][:-1]

        logName = "log." + application
        idStr = "End"

        # check whether the computation began
        if not os.path.isfile(self.caseDir + logName):
            return False

        # check whethet it ended
        isInLog = self.findInLog(logName, idStr)

        if isInLog == None:
            return False

        else:
            return True

    # read
    def readCSVFile(self, fileName):
        fileName = self.caseDir + fileName
        with open(fileName, 'r') as file:
            reader = csv.reader(file)

            cols = next(reader)

            data = list()
            for line in reader:
                data.append([float(i) for i in line])

        data = np.array(data)

        return cols, data

    # run
    def runFoamApp(self, apps):
        auxRun = self.caseDir + "auxRun.sh"
        with open(auxRun, 'w') as file:
            file.write("#!/bin/bash\n")
            file.write("cd $(dirname $0)\n")
            file.write("source /opt/" + self.foamToLoad + "/etc/bashrc\n")
            file.write(". $WM_PROJECT_DIR/bin/tools/RunFunctions\n")

            for app in apps:
                file.write("runApplication " + app + "\n")

        if self.devnull:
            subprocess.call(["bash", auxRun], stdout = subprocess.DEVNULL)

        else:
            log = open(self.logDir + "log.auxRun", 'w')
            subprocess.call(["bash", auxRun], stdout = log)

            log.close()

        os.remove(auxRun)
        os.remove(self.caseDir + "log." + app.split()[0])

    # geometry
    def readGeometry(self):
        geomDict = {}

        fileName = self.caseDir + "README"
        with open(fileName, 'r') as file:
            line = file.readline()

            while line:
                if "geometrical parameters" in line:
                    line = file.readline()

                    while line:
                        line = line.split(" = ")
                        try:
                            geomDict[line[0]] = float(line[1])

                        except:
                            break

                        line = file.readline()

                line = file.readline()

        return geomDict

    # logs
    def findInLog(self, logName, idStr):
        # logName - name of a log file I want to search in
        # idStr - string specifing a line containing the wanted thing

        # read log
        fileName = self.caseDir + logName
        with open(fileName,'r') as file:
            data = file.readlines()

        toReturn = None

        for line in data:
            if idStr in line:
                toReturn = line[:-1]

        return toReturn

    # info
    def numberOfCells(self):
        logName = "log.blockMesh"

        nCells = self.findInLog(logName, "nCells")
        nCells = int(nCells.split()[-1])

        return nCells

    # field average
    def fieldAverageOnPatch(self, field, patchName, fileName = None, latestTime = True, overwrite = False):
        dirName = self.caseDir + "postProcessing/patchAverage(" + field + ",name=" + patchName + ")/"
        if overwrite and os.path.isdir(dirName):
            os.rmtree(dirName)

        if not os.path.isdir(dirName):
            if latestTime:
                app = "postProcess -func \"patchAverage(" + field + ", name = " + patchName + ")\" -latestTime"

            else:
                app = "postProcess -func \"patchAverage(" + field + ", name = " + patchName + ")\""

            apps = [app]

            self.runFoamApp(apps)

        fileName = glob.glob(dirName + "*/surfaceFieldValue.dat")[0]
        with open(fileName, 'r') as file:
            data = file.readlines()
    
        value = float(data[-1].split()[-1])

        return value

    # flow rate
    def flowRateThroughPatch(self, patchName, fileName = None, latestTime = True, overwrite = False):
        dirName = self.caseDir + "postProcessing/flowRatePatch(name=" + patchName + ")/"
        if overwrite and os.path.isdir(dirName):
            os.rmtree(dirName)

        if not os.path.isdir(dirName):
            if latestTime:
                app = "postProcess -func \"flowRatePatch(name = " + patchName + ")\" -latestTime"

            else:
                app = "postProcess -func \"flowRatePatch(name = " + patchName + ")\""

            apps = [app]

            self.runFoamApp(apps)

        fileName = glob.glob(dirName + "*/surfaceFieldValue.dat")[0]
        with open(fileName, 'r') as file:
            data = file.readlines()
    
        value = float(data[-1].split()[-1])

        return value
