#!/usr/bin/python

#FILE DESCRIPTION=======================================================
# Python script used for an automated case construction
#
# This is the variant of the script for the automatic creations of the
# steady state RANS simulations for the liquid-liquid flow in ejector
#
# used solver is simpleFoam
#
# Notes:
# - all the "constant" directory is directly copied from the base case
# - all the "system" directory is directly copied from the base case
#   (including fvSchemes, fvSolution and controlDict)
# - the only changed thing are the boundary conditions and geometry
#
#
   
#LICENSE================================================================
#  caseConstructor.py
#
#  Copyright 2015-2019 Martin Isoz <martin@Poctar>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

#########DO NOT EDIT####################################################

#IMPORT BLOCK===========================================================
import os
import math
import io
import sys
import numpy as np
import shutil as sh

#IMPORT BLOCK-CUSTOM====================================================
# custom functions------------------------------------------------------
geomGenerator = "fblockMeshDictGen"

importCommand = "from " + geomGenerator + " import *"
exec(importCommand)
geomGenerator   += ".py"

#########EDITABLE#######################################################

#INPUT PARAMETERS=======================================================
# -- case defining parameters
# flow rate through nozzle, m3/s
QInLst = [0.3e-3, 0.4e-3, 0.5e-3]
pSucLst = [81.325, 81.325, 81.325]

# TURBULENCE INTENSITIES AT INLET/SUCTION
I0Inl   = 0.01
I0Suc   = 0.01
# Note: these are fairly low values, based on pipes

# -- geometry and meshing parameters
# nozzle dimensions
DNz1	= 0.008*2
DNz2	= 0.0043
DNz2O	= 0.0125*2
AngNz	= 40
    
LNz23	= 0.0224
LNz1	= 0.06-LNz23

# suction chamber length
LSuc = 0.060

# gap dimensions
LGap	= 0.015
WGap    = 0.0175*2
    
# converging part dimensions
LConv	= 0.02
WConv	= 0.0175*2

# mixing tube dimensions
LMxT	= 0.2
WMxT    = 0.014

# diffusor dimensions
LDiff	= 0.13
WDiff	= 0.0178*2

# things for edge, function from auxiliarFuncs.py
edgeFunction	= "polyLine"
splineType	= "polyLine"
convCPs = [[0.5,0.5]]
diffuserCPs = [[0.5,0.5]]
edgeSkip	= 0.002

# post-diffuser part dimensions
LPost	= 0.025
WPost   = 0.0175*2

geomOrig = [0.0,0.0,0.0]                                                 #geometry origin (middle of the inlet)
cSX = 1.0
cSY = 0.5
cellSize	= [cSX*1e-3, cSY*1e-3, 0.001]

# -- case run properties - of.sh file
nCores	= 2
startTime   = 0                                                         #simulation startTime
endTime	= 5000
wrInt       = endTime                                                      #simulation write interval
queName     = "Mshort"                                                   #que name for kraken
wallTime    = "2-00:00:00"                                               #walltime in hours for kraken
nameSpecs   = []
specFolder  = "ohneDent/"

# -- case specification
caseID = ""
genDir	= "/home/lumiforos/Workbench/Aquarium/Leviathan/caseConstructors/caseConstructorEjectorHorizontal/"
baseCase = genDir + "10_baseCaseFixedP"
baseDir	= genDir + "ZZ_cases/" + specFolder

for q in range(len(QInLst)):
    QIn = QInLst[q]
    pSuc = pSucLst[q]

    caseDir     = (baseDir + "sFEj_QIn" + 
                repr(round(QIn*1e3,4)) + "_pSuc" + 
                repr(round(pSuc,4)) + "_genCase"
                )

    for nameSpec in nameSpecs:
        caseDir += "_" + nameSpec + "_" + str(eval(nameSpec))

    caseDir += "_" + str(caseID) + "/"

    geomPars = [
        [DNz1,DNz2,DNz2O,AngNz,LNz1,LNz23],
        [LSuc,LGap,WGap],
        [LConv,WConv],
        [LMxT,WMxT],
        [LDiff,WDiff],
        [LPost,WPost],
        [edgeFunction, splineType, convCPs, diffuserCPs, edgeSkip],
    ]   

    #########PREFERABLY DO NOT EDIT#########################################
    
    #COPY CASE BASICS FROM THE BASECASE=====================================
    if os.path.isdir(caseDir):                                              #ensure, that the caseDir is clear
        sh.rmtree(caseDir)
        
    sh.copytree(baseCase,caseDir)                                             #copy data to caseDir
    
    #GENERATE BLOCKMESHDICT (CASE GEOMETRY)=================================
    blockMeshClass, badMesh = genBlockMeshDict(geomPars,geomOrig,cellSize,caseDir)
    
    #SPECIFY CURRENT SCRIPT VERSIONS========================================
    blockMeshClass  += ".py"
    caseConstructor = os.path.basename(__file__)
   
    #COPY CURRENT SCRIPT VERSIONS===========================================
    scFolder= genDir + "/01_pyCodes/"                                              #folder with Scripts sources
    scNames = [ 
                geomGenerator,
                blockMeshClass,
                caseConstructor,
    ]                                   
    for scName in scNames:
        sh.copyfile(scFolder + scName,caseDir + scName)     #copy current script version
    
    #CASE CONSTANTS AND CALCULATIONS========================================
    # input data------------------------------------------------------------
    # -- global parameters
    g       = 9.81                                                          #grav. acc., m/s2
    
    # -- liquid properties
    #~ rhoL,muL= 997.0,0.8905e-3                                           #e-tabulky
    rhoL,muL=1000.0,1.0e-3                                              #"fresh" water
    nu      = muL/rhoL                                                  #kinematic viscosity
    
    # ESTIMATE TURBULENCE VARIABLES
    uNzl = QIn/(math.pi*(DNz1*0.5)**2.0)
    
    # further calculations
    Re = uNzl*WConv/nu
    Cmu= 0.09
    
    k0Inl = 1.5*uNzl**2.0*I0Inl**2.0
    lScInl= (DNz1*0.5)*2.0
    e0Inl = Cmu*k0Inl**1.5/lScInl
    w0Inl = e0Inl/(Cmu*k0Inl)
    nuTInl= k0Inl/w0Inl
    
    uSuc=QIn/(math.pi*((WGap/2.0)**2-(DNz2O/2.0)**2))
    k0Suc = 1.5*uSuc**2.0*I0Suc**2.0
    lScSuc= (WGap-DNz2O)/2.0
    e0Suc = Cmu*k0Suc**1.5/lScSuc
    w0Suc = e0Suc/(Cmu*k0Suc)
    nuTSuc= k0Suc/w0Suc

    uI = uNzl                                       #all is approximative
    
    #OPEN AUTOGENERATED README FILE=========================================
    README  = open(caseDir + "./README",'a')                                #open file to append
    README.write("\ncaseDir:" + caseDir + "\n\n")
    # -- start by writing basic case info and geometry
    README.write("\ngeometrical parameters [m]\n")
    geomPars = ["DNz1", "DNz2", "DNz2O", "AngNz", "LNz1", "LNz23", "LConv", "WConv", "LMxT", "WMxT", "LDiff", "WDiff", "LPost", "WPost"]

    for i in geomPars:
        README.write(i + " = " + str(eval(i)) + "\n")

    for pID, point in enumerate(convCPs):
        README.write("xC" + str(pID + 1) + " = " + str(point[0]) + "\n")
        README.write("yC" + str(pID + 1) + " = " + str(point[1]) + "\n")

    for pID, point in enumerate(diffuserCPs):
        README.write("xD" + str(pID + 1) + " = " + str(point[0]) + "\n")
        README.write("yD" + str(pID + 1) + " = " + str(point[1]) + "\n")

    README.write("\nother ineteresting facts\n")
    README.write("cellSize  \t = \t " + repr(cellSize) + " m\n")
    README.write("rho       \t = \t " + repr(rhoL) + " kgm-3\n")
    README.write("nu        \t = \t " + repr(nu) + " m2s-1\n")
    README.write("ReInl     \t = \t " + repr(round(Re,4)) + "\n")
    README.write("k0Inl     \t = \t %.4g"%(k0Inl) + " Jkg-1\n")
    README.write("w0Inl     \t = \t %.4g"%(w0Inl) + " s-1\n")
    README.write("e0Inl     \t = \t %.4g"%(e0Inl) + " s-1\n")
    README.write("k0Suc     \t = \t %.4g"%(k0Suc) + " Jkg-1\n")
    README.write("w0Suc     \t = \t %.4g"%(w0Suc) + " s-1\n")
    README.write("e0Suc     \t = \t %.4g"%(e0Suc) + " s-1\n")
    README.write("nCores    \t = \t " + repr(nCores) + "\n")
    README.write("startTime \t = \t " + repr(startTime) + " s\n")
    README.write("endTime   \t = \t " + repr(endTime) + " s\n")
    
    #BC FILES MODIFICATION==================================================
    #-----------------------------------------------------------------------
    # FUNCTION CALL
    #-----------------------------------------------------------------------
    print ("ADJUSTING BC===============================\n\n")
    README.write("\n 0.org==============================================\n")
    #-----------------------------------------------------------------------
    # U
    #-----------------------------------------------------------------------
    #
    # Boundary conditions for the velocity field
    #
    README.write("\n U\n")
    
    pVals   = ["uniform (" + repr(uI) + " 0 0)"]           #inlet liquid velocity speed
    
    idStr   = ["uniform (uI 0 0)"]
    
    # write everything to the file
    with open(caseDir + "./0.org/U", "r") as file:
        # read a list of lines into data
        data = file.readlines()
        
    for j in range(len(idStr)):
        for i in range(len(data)):
            fInd = data[i].find(idStr[j])
            if fInd>-1:
                data[i] = data[i][:fInd] + pVals[j] + ";\n"
    
    with open(caseDir + "./0.org/U", "w") as file:
        file.writelines( data )
        README.writelines( data )                                           #write to readme
        
    print ("DONE=======================================\n\n")
    
    #-----------------------------------------------------------------------
    # p
    #-----------------------------------------------------------------------
    #
    # Boundary conditions for the pressure field
    #
    README.write("\n p\n")
    
    pVals   = ["uniform " + repr(pSuc)]
   
    idStr   = ["uniform pSuc"]
    
    # write everything to the file
    with open(caseDir + "./0.org/p", "r") as file:
        # read a list of lines into data
        data = file.readlines()
        
    for j in range(len(idStr)):
        for i in range(len(data)):
            fInd = data[i].find(idStr[j])
            if fInd>-1:
                data[i] = data[i][:fInd] + pVals[j] + ";\n"
    
    with open(caseDir + "./0.org/p", "w") as file:
        file.writelines( data )
        README.writelines( data )                                           #write to readme
        
    print ("DONE=======================================\n\n")
    
    #-----------------------------------------------------------------------
    # k, omega and epsilon
    #-----------------------------------------------------------------------
    #
    # Boundary conditions for turbulent variables
    #
    varNames = ["k","omega","epsilon"]
    varVals  = [
        [k0Inl,k0Suc],
        [w0Inl,w0Suc],
        [e0Inl,e0Suc],
    ]
    
    for varInd in range(len(varNames)):
        varNm   = varNames[varInd]
        var0Inl,var0Suc = varVals[varInd]
        
        README.write("\n %s\n"%(varNm))
        
        pVals   = [
            "internalField   uniform " + repr(var0Suc),
            "value uniform " + repr(var0Suc),
            "inletValue uniform " + repr(var0Suc),
            "value uniform " + repr(var0Suc),
        ]
        
        idStr   = [
            "internalField   uniform var0",
            "value           inletValueInlet",
            "inletValue      inletValueSuction",
            "value           inletValueSuction",
        ]
        
        # write everything to the file
        with open(caseDir + "./0.org/%s"%(varNm), "r") as file:
            # read a list of lines into data
            data = file.readlines()
            
        for j in range(len(idStr)):
            for i in range(len(data)):
                fInd = data[i].find(idStr[j])
                if fInd>-1:
                    data[i] = data[i][:fInd] + pVals[j] + ";\n"
        
        with open(caseDir + "./0.org/%s"%(varNm), "w") as file:
            file.writelines( data )
            README.writelines( data )                                        #write to readme
            
    #-----------------------------------------------------------------------
    # nut
    #-----------------------------------------------------------------------
    #
    # turbulence kinematic viscosity initial guess
    #
    README.write("\n nut\n")
    
    pVals   = ["uniform " + repr(nuTInl*1e-3)]           #inlet liquid velocity speed
    
    idStr   = ["uniform 0"]
    
    # write everything to the file
    with open(caseDir + "./0.org/nut", "r") as file:
        # read a list of lines into data
        data = file.readlines()
        
    for j in range(len(idStr)):
        for i in range(len(data)):
            fInd = data[i].find(idStr[j])
            if fInd>-1:
                data[i] = data[i][:fInd] + pVals[j] + ";\n"
    
    with open(caseDir + "./0.org/nut", "w") as file:
        file.writelines( data )
        README.writelines( data )                                           #write to readme
        
    print ("DONE=======================================\n\n")
    
    #CONSTANTS DIRECTORY FILES MODIFICATIONS================================
    print ("ADJUSTING FILES IN ./CONSTANTS=============\n\n")
    README.write("\n CONSTANTS==========================================\n")
    README.write("\n transportProperties\n")
    
    idStr = [
        "nu              [0 2 -1 0 0 0 0]",
    ]         
    
    pVals = [[nu]]
    
    # write everything to the file
    with open(caseDir + "./constant/transportProperties", "r") as file:
        # read a list of lines into data
        data = file.readlines()
        
    for j in range(len(idStr)):
        k = 0
        for i in range(len(data)):
            fInd = data[i].find(idStr[j])
            if fInd>-1:
                data[i] = data[i][:fInd] + idStr[j] + "\t" + repr(pVals[j][k]) + ";\n"
                k = k+1
    
    with open(caseDir + "./constant/transportProperties", "w") as file:
        file.writelines( data )
        README.writelines( data )                                       #write to readme
        
    print ("DONE=======================================\n\n")
    #SYSTEM DIRECTORY FILES MODIFICATIONS===================================
    print ("ADJUSTING FILES IN ./SYSTEM================\n\n")
    README.write("\n SYSTEM=============================================\n")
        
    #-----------------------------------------------------------------------
    # decomposeParDict
    #-----------------------------------------------------------------------
    #
    # decomposes the case for run on multiple cores
    #
    README.write("\n decomposeParDict\n")
    
    idStr = ["numberOfSubdomains "]
    
    pVals = [repr(nCores)]
    
    # write everything to the file
    with open(caseDir + "./system/decomposeParDict", "r") as file:
        # read a list of lines into data
        data = file.readlines()
        
    for j in range(len(idStr)):
        for i in range(len(data)):
            fInd = data[i].find(idStr[j])
            if fInd>-1:
                data[i] = data[i][:fInd] + idStr[j] + "\t" + pVals[j] + ";\n"
    
    with open(caseDir + "./system/decomposeParDict", "w") as file:
        file.writelines( data )
        README.writelines( data )                                           #write to readme

    #-----------------------------------------------------------------------
    # controlDict
    #-----------------------------------------------------------------------
    #
    # creates initial condition for the case (presence of the liquid)
    #
    README.write("\n controlDict\n")
    
    idStr = ["startTime ","endTime ","writeInterval "]
    
    pVals = [repr(startTime),repr(endTime),repr(wrInt)]
    
    # write everything to the file
    with open(caseDir + "./system/controlDict", "r") as file:
        # read a list of lines into data
        data = file.readlines()
        
    for j in range(len(idStr)):
        for i in range(len(data)):
            fInd = data[i].find(idStr[j])
            if fInd>-1:
                data[i] = data[i][:fInd] + idStr[j] + "\t" + pVals[j] + ";\n"
    
    with open(caseDir + "./system/controlDict", "w") as file:
        file.writelines( data )
        README.writelines( data )                                           #write to readme
        
        
    print ("DONE=======================================\n\n")
    #CLOSE THE AUTOGENERATED README FILE====================================
    README.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * // \n\n")
    README.close()

    # TO CATCH REALLY BAD MESH WHICH WOULD KILL THE BLOCKMESH
    if badMesh:
        with open(caseDir + "./log.blockMesh", "w") as file:
            file.write("Failed due to really bad mesh")
