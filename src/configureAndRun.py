
import os
import sys
import subprocess
import shutil as sh

# auxiliary functions for generetion of cases
class configureAndRun:
    def __init__(self, genDir = None, caseConstructor = None, baseCase = None, paraviewDir = None, caseID = "", logDir = "./logs", devnull = True):
        # genDir - path to the case constructing directory
        # caseConstructor - current caseConstructor*.py name
        # baseCase - path to the baseCase directory
        # paraviewDir - path to the directory where files used by paraview are kept
        # caseID - unique number representing the case

        self.genDir = genDir
        self.caseConstructor = caseConstructor
        self.baseCase = baseCase
        self.paraviewDir = paraviewDir
        self.caseID = caseID
        self.logDir = logDir
        self.devnull = devnull

        if not os.path.exists(logDir) and not self.devnull:
            os.makedirs(logDir)

    def changeParsInCaseConstructor(self, pars, values):
        # pars - list of names of parameters to change
        # values - list of values to change the parametrs to, indexes must correspond

        if not self.caseID == "":
            pars.append("caseID")
            values.append(self.caseID)
    
        with open(self.genDir + "01_pyCodes/" + self.caseConstructor,'r') as file:
            data = file.readlines()
    
        for id, par in enumerate(pars):
            toFind = str(par)
    
            for i in range(0,len(data)):
                found = data[i].find(toFind)
                if found == 0 and "=" in data[i]:
                    if isinstance(values[id], str):
                        data[i] = toFind + "\t= \""+str(values[id])+"\"\n"
                        break

                    else:
                        data[i] = toFind + "\t= "+str(values[id])+"\n"
                        break
    
        if not len(data) == 0:
            with open(self.genDir + "01_pyCodes/" + self.caseConstructor.split(".py")[0] + "_" + str(self.caseID) + ".py",'w') as file:
                file.writelines(data)

    def changeThingsInAllrun(self, pars, values, Allrun):
        # Allrun -  name of the Allrun bash script

        with open(self.baseCase + Allrun,'r') as file:
            data = file.readlines()
    
        for id, par in enumerate(pars):
            toFind = str(par)
    
            for i in range(0,len(data)):
                found = data[i].find(toFind)
                if found > -1:
                    if isinstance(values[id], str):
                        data[i] = toFind + "=\""+str(values[id])+"\"\n"
                        break

                    else:
                        data[i] = toFind + "="+str(values[id])+"\n"
                        break
    
        with open(self.baseCase + Allrun + "_" + str(self.caseID),'w') as file:
            file.writelines(data)

    def changeThingsInAutoParaviewer(self, pars, values, autoParaviewer):
        # pars - list of names of parameters to change
        # values - list of values to change the parametrs to, indexes must correspond
        # autoParaviewer - name of the python script to change

        with open(self.paraviewDir + autoParaviewer,'r') as file:
            data = file.readlines()
    
        for id, par in enumerate(pars):
            toFind = str(par)
    
            for i in range(0,len(data)):
                found = data[i].find(toFind)
                if found > -1:
                    if isinstance(values[id], str):
                        data[i] = toFind + "\t= \""+str(values[id])+"\"\n"
                        break

                    else:
                        data[i] = toFind + "\t= "+str(values[id])+"\n"
                        break
    
        with open(self.baseCase + autoParaviewer,'w') as file:
            file.writelines(data)

    def makeCase(self, changeDict):
        # changeDict - dictionary; name of function from configureAndRun to use:list of its arguments 

        for function in changeDict.keys():
            argLst = changeDict.get(function)

            command = "self." + function + "("
            for arg in argLst:
                if isinstance(arg, str):
                    command += "\"" + str(arg) + "\", "
                else:
                    command += str(arg) + ", "
    
            if len(argLst)>0:
                command = command[:-2] + ")"
            else:
                command = command + ")"

            exec(command)
    
            # run caseConstructor
            if self.devnull:
                subprocess.call(["python", self.genDir + "01_pyCodes/" + self.caseConstructor.split(".py")[0] + "_" + str(self.caseID) + ".py"], stdout = subprocess.DEVNULL)

            else:
                log = open("./logs/log.caseConstructor" + "_" + str(self.caseID),'w')
                subprocess.call(["python", self.genDir + "01_pyCodes/" + self.caseConstructor.split(".py")[0] + "_" + str(self.caseID) + ".py"], stdout = log)

                log.close()

        if not self.caseID == "":
            os.remove(self.genDir + "01_pyCodes/" + self.caseConstructor.split(".py")[0] + "_" + str(self.caseID) + ".py")
    
    def runSingleCase(self, caseDir, Allrun):
        target = caseDir + Allrun

        if self.devnull:
            subprocess.call(["bash", target], stdout = subprocess.DEVNULL)

        else:
            log = open("./logs/log." + Allrun + "_" + str(self.caseID),'w')
            subprocess.call(["bash", target], stdout = log)

            log.close()

    def writeDictionaryForFoam(self, toWrite, caseDir, targetDir, name):
        # toWrite - a list of lists, each list beginns with a type (list, None), consecutive members are type specific

        fileName = caseDir + targetDir + name
        with open(fileName, 'w') as file:
            # write FOAM header
            file.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            file.write("| =========                 |                                                 |\n")
            file.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            file.write("|  \\\\    /   O peration     | Version:  3.0.1                                 |\n")
            file.write("|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n")
            file.write("|    \\\\/     M anipulation  |                                                 |\n")
            file.write("\*---------------------------------------------------------------------------*/\n")
            file.write("FoamFile\n")
            file.write("{\n")
            file.write("    version     2.0;\n")
            file.write("    format      ascii;\n")
            file.write("    class       dictionary;\n")
            file.write("    object      " + name + ";\n")
            file.write("}\n")
            file.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")
            file.write("\n")

            # write the actual content
            for part in toWrite:
                partType = part[0]

                if partType == "list":
                    file.write(part[1] + "\n")
                    file.write("(\n")
                    file.write("\n".join(["\t"+str(i) for i in part[2]]) + "\n")
                    file.write(");\n")

                elif partType == "scalar":
                    file.write(str(part[1]) + "\t" + str(part[2]) + ";\n")

                elif partType == None:
                    file.write("\n".join(part[1]))

            # write FOAM ending
            file.write("\n")
            file.write("// ************************************************************************* //")

    def writePostProcessing(self, apps, caseDir, Allrun):
        # apps - list of all apps required to be run
        # Allrun - name of the Allrun bash script

        fileName = caseDir + Allrun
        with open(fileName, 'a') as file:
            file.write("### autowritten postProcessing\n")
            for utility in apps:
                file.write("runApplication " + utility + "\n")
