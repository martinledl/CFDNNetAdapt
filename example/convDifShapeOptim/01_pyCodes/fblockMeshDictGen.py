#!/usr/bin/python

#FILE DESCRIPTION=======================================================
# a function to generate blockMeshDict for simple ejector nozzle-diffuser
# geometry
#
# The output is a simple blockMeshDict to generate the mesh
#
# axial symmetry - wedge
# diffuser control points - spline edge


#LICENSE================================================================
#  blockMeshDictGen.py
#  
#  Copyright 2018-2019 Martin Isoz & Lucie Kubickova
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

#IMPORT BLOCK=======================================================
import os
import re
import sys
import copy
import math

from interpolationFuncs import *

blockMeshClass = "blockMeshDictClassV8"
command = "from " + blockMeshClass + " import *"
exec(command)

#===================================================================
#							EDITABLE
#===================================================================
def genBlockMeshDict(
    geomPars,
    geomOrig,
    cellSize,
    caseDir,
):
    #GET THE DATA=======================================================
    #-------------------------------------------------------------------
    # GEOMETRY DATA
    #-------------------------------------------------------------------
    
    # -- unpack the input parameters
    nzPars,gapPars,convPars,mxTPars,dfPars,postPars,splinePars=geomPars
    
    DNz1,DNz2,DNz2O,AngNz,LNz1,LNz23   = nzPars
    LSuc,LGap,WGap                   = gapPars
    LConv,WConv                 = convPars
    LMxT,WMxT                   = mxTPars
    LDiff,WDiff                 = dfPars
    LPost,WPost                 = postPars
    edgeFunction,splineType,convCPs,diffuserCPs,edgeSkip     = splinePars

    x0,y0,z0 = geomOrig
    
    # -- auxiliary data
    WWall2        = (DNz2O-DNz2)*0.5

    R1 = WConv*0.5
    A1 = DNz2*0.5
    B1 = WWall2
    C1 = (WConv-DNz2)*0.5-WWall2
    
    R2 = WMxT*0.5
    A2 = A1*R2/R1
    B2 = B1*R2/R1
    C2 = C1*R2/R1
    
    R3 = WMxT*0.5
    A3 = A1*R3/R1
    B3 = B1*R3/R1
    C3 = C1*R3/R1

    R4 = WDiff*0.5
    A4 = A1*R4/R1
    B4 = B1*R4/R1
    C4 = C1*R4/R1
    
    # -- compute necessary dimensions
    # Note: this is mostly a conversion of absolute dimensions to the
    #       relative ones used during the geometry construction
    RNz1,RNz2 = 0.5*DNz1,0.5*DNz2
    RGap = 0.5*WGap
    RNz2O = 0.5*DNz2O
    AngNz  = 0.5*AngNz/180.0*math.pi
    LNz2 = (RNz1-RNz2)/math.tan(AngNz)
    
    LNz3 = LNz23 - LNz2

    alpha = math.atan((DNz1-DNz2)*0.5/LNz2)
    radius = edgeSkip/math.tan(alpha/2)
    hypotenuse = math.sqrt(radius**2 + edgeSkip**2)
    edgeY1 = (hypotenuse-radius)*radius/hypotenuse
    
    alpha = math.atan((WGap-WMxT)*0.5/LConv)
    radius = edgeSkip/math.tan(alpha/2)
    hypotenuse = math.sqrt(radius**2 + edgeSkip**2)
    edgeY2 = (hypotenuse-radius)*radius/hypotenuse
    
    alpha = math.atan((WDiff-WMxT)*0.5/LDiff)
    radius = edgeSkip/math.tan(alpha/2)
    hypotenuse = math.sqrt(radius**2 + edgeSkip**2)
    edgeY3 = (hypotenuse-radius)*radius/hypotenuse

    RNz12 = edgeSkip/LNz2*(RNz1-RNz2) + RNz2
    LNz2 += edgeSkip
    LNz3 -= edgeSkip

    WGM = (edgeSkip/LConv*(WGap-WMxT) + WMxT)*0.5
    LConv += edgeSkip
    LMxT -= edgeSkip
    
    LMxT += edgeSkip
    LDiff -= edgeSkip

    yMax = 0.066

    #-------------------------------------------------------------------
    # MESH DATA
    #-------------------------------------------------------------------
    
    # size of a single cell
    dX,dY,dZ = cellSize
    
    # general Y dir. dicretization (all behind nozzle body)
    factorY = 2
    nCAY = int(round(A1/dY))
    nCBY = int(round(B1/dY))
    nCCY = int(round(C1/dY))*factorY

    # mesh grading (region refinements)
    grXConv = "%g"%(1/2.2)
    grXMxT  = "((0.3 0.4 3.0)(0.3 0.2 1.0)(0.4 0.4 %g))"%(1/3.0)
    grXPost = "2.4"

    grYB = "1.0"
    grYC = "%g"%(0.33/factorY)
    grYD = "1.0"

    grXSuc = "%g"%(1/7.0)
    grXFront = "%g"%(1/3.0)
    grXBack = "3.0"
    
    # mesh grading (basic)
    grX, grY, grZ = "1.0", "1.0", "1.0"

    # mesh scale
    mScale  = 1

    # computations for converging part edge
    convPnts = convCPs[:] # do not dare to put this away
    convPnts.insert(0,[0.0,1.0])
    convPnts.append([1.0,0.0])

    NConv = int(round(LConv/dX))

    command = edgeFunction + "(convPnts, NConv)"
    convPnts = eval(command)

    # computations for diffuser edge
    diffuserPnts = diffuserCPs[:] # do not dare to put this away
    diffuserPnts.insert(0,[0.0,0.0])
    diffuserPnts.append([1.0,1.0])

    NDiff = int(round(LDiff/dX))

    command = edgeFunction + "(diffuserPnts, NDiff)"
    diffuserPnts = eval(command)

    # switch
    badMesh = False

    #===================================================================
    #							DO NOT EDIT
    #===================================================================
    
    #-------------------------------------------------------------------
    # GENERATE THE BLOCK OBJECTS
    #-------------------------------------------------------------------
    
    fvMesh = mesh()
    
    #-----------------------------------------------------------------------
    # - nozzle body
    # nozzleBody
    xC,yC,zC = x0,y0,z0
    xE,yE = xC+LNz1,yC+RNz1

    # vertex coordinates
    vertices = [
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE,zC-yE/yMax*dZ],
            [xC,yE,zC-yE/yMax*dZ],
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE,zC+yE/yMax*dZ],
            [xC,yE,zC+yE/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = []

    # number of cells
    nCX = int(round((xE-xC)/dX))
    nCells = [nCX, nCAY, 1]

    # grading
    grading = [grX, grY, grZ]

    # create the block
    nozzleBody = fvMesh.addBlock(vertices, neighbors, nCells, grading)
    
    #-----------------------------------------------------------------------
    # - nozzle outlet area
    # nozzleConv
    xC,yC,zC = x0+LNz1,y0,z0
    xE,yE01,yE02 = xC+LNz2,yC+RNz1,yC+RNz2

    # vertex coordinates
    vertices = [
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [nozzleBody]

    # number of cells
    nCX = int(round((xE-xC)/dX))
    nCells = [nCX,nCAY,1]

    # grading
    grading = [grX,grY,grZ]

    # create the block
    nozzleConv = fvMesh.addBlock(vertices, neighbors, nCells, grading)
    
    # define special edges
    pnts = [[xE-2*edgeSkip,yC+RNz12],[xE-edgeSkip,yE02+edgeY1],[xE,yE02]]
    N = int(round(2*edgeSkip/dX))

    arcPnts = arcOf3Pnts(pnts,N)

    e32 = list()
    e76 = list()
    for point in arcPnts:
        e32.append((point[0], point[1], zC-point[1]/yMax*dZ))
        e76.append((point[0], point[1], zC+point[1]/yMax*dZ))

    fvMesh.addEdge("polyLine", nozzleConv.retEYEZ0(), e32)
    fvMesh.addEdge("polyLine", nozzleConv.retEYEZE(), e76)
    
    #-----------------------------------------------------------------------
    # - nozzle tube area
    # nozzleTube
    xC,yC,zC = xE,y0,z0
    xE,yE = xC+LNz3,yE02

    # vertex coordinates
    vertices = [
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE,zC-yE/yMax*dZ],
            [xC,yE,zC-yE/yMax*dZ],
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE,zC+yE/yMax*dZ],
            [xC,yE,zC+yE/yMax*dZ],
        ]
    
    # neighboring blocks
    neighbors = [nozzleConv]
    
    # number of cells
    nCX = int(round((xE-xC)/dX))*2
    nCells = [nCX,nCAY,1]
    
    # grading
    grading = [grXFront,grY,grZ]
    
    # create the block
    nozzleTube = fvMesh.addBlock(vertices, neighbors, nCells, grading)
    
    #-----------------------------------------------------------------------
    # suction chamber
    xC,yC,zC = x0+LNz1+LNz23-LSuc,y0+RNz2O,z0
    xE,yE = xC+LSuc,y0+R1

    # vertex coordinates
    vertices = [
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE,zC-yE/yMax*dZ],
            [xC,yE,zC-yE/yMax*dZ],
            [xC,yC,zC+yC/yMax*dZ],
            [xE,yC,zC+yC/yMax*dZ],
            [xE,yE,zC+yE/yMax*dZ],
            [xC,yE,zC+yE/yMax*dZ],
        ]
    
    # neighboring blocks
    neighbors = []
    
    # number of cells
    nCX = int(round((xE-xC)/dX)*1.1)
    nCells = [nCX,nCCY,1]
    
    # grading
    grading = [grXSuc,grYC,grZ]
    
    # create the block
    suctionChamber = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    #-----------------------------------------------------------------------
    # - gap part
    # AGap
    AC,BC,CC = A1,B1,C1
    AN,BN,CN = A1,B1,C1

    xC,yC,zC = xE,y0,z0
    xE,yE01,yE02 = xC+LGap,yC+AC,yC+AN

    # vertex coordinates
    vertices = [
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [nozzleTube]
    
    # number of cells
    nCX = int(round((xE-xC)/dX))*2
    nCells = [nCX,nCAY,1]

    # grading
    grading = [grXBack,grY,grZ]

    # create the block
    AGap = fvMesh.addBlock(vertices, neighbors, nCells, grading)
    
    # BGap
    xC,yC01,yC02,zC = xC,yE01,yE02,zC
    xE,yE01,yE02 = xE,yC01+BC,yC02+BN

    # vertex coordinates
    vertices = [
            [xC,yC01,zC-yC01/yMax*dZ],
            [xE,yC02,zC-yC02/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC01,zC+yC01/yMax*dZ],
            [xE,yC02,zC+yC02/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [AGap, suctionChamber]
    
    # number of cells
    nCells = [nCX,nCBY,1]

    # grading
    grading = [grXBack,grYB,grZ]

    # create the block
    BGap = fvMesh.addBlock(vertices, neighbors, nCells, grading)
    
    # CGap
    xC,yC01,yC02,zC = xC,yE01,yE02,zC
    xE,yE01,yE02 = xE,yC01+CC,yC02+CN

    # vertex coordinates
    vertices = [
            [xC,yC01,zC-yC01/yMax*dZ],
            [xE,yC02,zC-yC02/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC01,zC+yC01/yMax*dZ],
            [xE,yC02,zC+yC02/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [suctionChamber, BGap]
    
    # number of cells
    nCells = [nCX,nCCY,1]

    # grading
    grading = [grXBack,grYC,grZ]

    # create the block
    CGap = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    #-----------------------------------------------------------------------
    # - converging part
    # AConv
    AC,BC,CC = A1,B1,C1
    AN,BN,CN = A2,B2,C2

    xC,yC,zC = xE,y0,z0
    xE,yE01,yE02 = xC+LConv,yC+AC,yC+AN

    # vertex coordinates
    vertices = [
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [AGap]
    
    # number of cells
    nCX = int(round((xE-xC)/dX*1.6))
    nCells = [nCX,nCAY,1]

    # grading
    grading = [grXConv,grY,grZ]

    # create the block
    AConv = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    # define special edges
    e32 = list()
    e76 = list()

    for point in convPnts:
        why = (yE01-yE02)*point[1]+yE02
        if why < 0.0:
            badMesh = True

        e32.append(((xE-xC)*point[0]+xC, why, zC-((yE01-yE02)*point[1]+yE02)/yMax*dZ))
        e76.append(((xE-xC)*point[0]+xC, why, zC+((yE01-yE02)*point[1]+yE02)/yMax*dZ))

    fvMesh.addEdge("polyLine", AConv.retEYEZ0(), e32)
    fvMesh.addEdge("polyLine", AConv.retEYEZE(), e76)

    # BConv
    xC,yC01,yC02,zC = xC,yE01,yE02,zC
    xE,yE01,yE02 = xE,yC01+BC,yC02+BN

    # vertex coordinates
    vertices = [
            [xC,yC01,zC-yC01/yMax*dZ],
            [xE,yC02,zC-yC02/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC01,zC+yC01/yMax*dZ],
            [xE,yC02,zC+yC02/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [BGap, AConv]
    
    # number of cells
    nCells = [nCX,nCBY,1]

    # grading
    grading = [grXConv,grYB,grZ]

    # create the block
    BConv = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    # define special edges
    e32 = list()
    e76 = list()

    for point in convPnts:
        e32.append(((xE-xC)*point[0]+xC, (yE01-yE02)*point[1]+yE02, zC-((yE01-yE02)*point[1]+yE02)/yMax*dZ))
        e76.append(((xE-xC)*point[0]+xC, (yE01-yE02)*point[1]+yE02, zC+((yE01-yE02)*point[1]+yE02)/yMax*dZ))

    fvMesh.addEdge("polyLine", BConv.retEYEZ0(), e32)
    fvMesh.addEdge("polyLine", BConv.retEYEZE(), e76)
    
    # CGap
    xC,yC01,yC02,zC = xC,yE01,yE02,zC
    xE,yE01,yE02 = xE,yC01+CC,yC02+CN

    # vertex coordinates
    vertices = [
            [xC,yC01,zC-yC01/yMax*dZ],
            [xE,yC02,zC-yC02/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC01,zC+yC01/yMax*dZ],
            [xE,yC02,zC+yC02/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [CGap, BConv]
    
    # number of cells
    nCells = [nCX,nCCY,1]

    # grading
    grading = [grXConv,grYC,grZ]

    # create the block
    CConv = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    # define special edges
    e32 = list()
    e76 = list()

    for point in convPnts:
        e32.append(((xE-xC)*point[0]+xC, (yE01-yE02)*point[1]+yE02, zC-((yE01-yE02)*point[1]+yE02)/yMax*dZ))
        e76.append(((xE-xC)*point[0]+xC, (yE01-yE02)*point[1]+yE02, zC+((yE01-yE02)*point[1]+yE02)/yMax*dZ))

    fvMesh.addEdge("polyLine", CConv.retEYEZ0(), e32)
    fvMesh.addEdge("polyLine", CConv.retEYEZE(), e76)

    #-----------------------------------------------------------------------   
    # - mixing tube area
    # AMxT
    AC,BC,CC = A2,B2,C2
    AN,BN,CN = A3,B3,C3
    
    xC,yC,zC = xE,y0,z0
    xE,yE01,yE02 = xC+LMxT,yC+AC,yC+AN

    # vertex coordinates
    vertices = [
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [AConv]
    
    # number of cells
    nCX = int(round((xE-xC)/dX)*0.9)
    nCells = [nCX,nCAY,1]

    # grading
    grading = [grXMxT,grY,grZ]

    # create the block
    AMxT = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    # BMxT
    xC,yC01,yC02,zC = xC,yE01,yE02,zC
    xE,yE01,yE02 = xE,yC01+BC,yC02+BN

    # vertex coordinates
    vertices = [
            [xC,yC01,zC-yC01/yMax*dZ],
            [xE,yC02,zC-yC02/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC01,zC+yC01/yMax*dZ],
            [xE,yC02,zC+yC02/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [BConv, AMxT]
    
    # number of cells
    nCells = [nCX,nCBY,1]

    # grading
    grading = [grXMxT,grYB,grZ]

    # create the block
    BMxT = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    # CMxT
    xC,yC01,yC02,zC = xC,yE01,yE02,zC
    xE,yE01,yE02 = xE,yC01+CC,yC02+CN

    # vertex coordinates
    vertices = [
            [xC,yC01,zC-yC01/yMax*dZ],
            [xE,yC02,zC-yC02/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC01,zC+yC01/yMax*dZ],
            [xE,yC02,zC+yC02/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [CConv, BMxT]
    
    # number of cells
    nCells = [nCX,nCCY,1]

    # grading
    grading = [grXMxT,grYC,grZ]

    # create the block
    CMxT = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    #-----------------------------------------------------------------------
    # - diffuser area
    # ADiff
    AC,BC,CC = A3,B3,C3
    AN,BN,CN = A4,B4,C4

    xC,yC,zC = xE,y0,z0
    xE,yE01,yE02 = xC+LDiff,yC+AC,yC+AN

    # vertex coordinates
    vertices = [
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [AMxT]
    
    # number of cells
    nCX = int(round((xE-xC)/dX)*1.7)
    nCells = [nCX,nCAY,1]

    # grading
    grading = [grX,grY,grZ]

    # create the block
    ADiff = fvMesh.addBlock(vertices, neighbors, nCells, grading)
    
    # define special edges
    e32 = list()
    e76 = list()

    for point in diffuserPnts:
        why = (yE02-yE01)*point[1]+yE01
        if why < 0.0:
            badMesh = True

        e32.append(((xE-xC)*point[0]+xC, why, zC-((yE02-yE01)*point[1]+yE01)/yMax*dZ))
        e76.append(((xE-xC)*point[0]+xC, why, zC+((yE02-yE01)*point[1]+yE01)/yMax*dZ))

    fvMesh.addEdge("polyLine", ADiff.retEYEZ0(), e32)
    fvMesh.addEdge("polyLine", ADiff.retEYEZE(), e76)
    
    # BDiff
    xC,yC01,yC02,zC = xC,yE01,yE02,zC
    xE,yE01,yE02 = xE,yC01+BC,yC02+BN

    # vertex coordinates
    vertices = [
            [xC,yC01,zC-yC01/yMax*dZ],
            [xE,yC02,zC-yC02/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC01,zC+yC01/yMax*dZ],
            [xE,yC02,zC+yC02/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [BMxT, ADiff]
    
    # number of cells
    nCells = [nCX,nCBY,1]

    # grading
    grading = [grX,grYB,grZ]

    # create the block
    BDiff = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    # define special edges
    e32 = list()
    e76 = list()

    for point in diffuserPnts:
        e32.append(((xE-xC)*point[0]+xC, (yE02-yE01)*point[1]+yE01, zC-((yE02-yE01)*point[1]+yE01)/yMax*dZ))
        e76.append(((xE-xC)*point[0]+xC, (yE02-yE01)*point[1]+yE01, zC+((yE02-yE01)*point[1]+yE01)/yMax*dZ))

    fvMesh.addEdge("polyLine", BDiff.retEYEZ0(), e32)
    fvMesh.addEdge("polyLine", BDiff.retEYEZE(), e76)
    
    # CDiff
    xC,yC01,yC02,zC = xC,yE01,yE02,zC
    xE,yE01,yE02 = xE,yC01+CC,yC02+CN

    # vertex coordinates
    vertices = [
            [xC,yC01,zC-yC01/yMax*dZ],
            [xE,yC02,zC-yC02/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC01,zC+yC01/yMax*dZ],
            [xE,yC02,zC+yC02/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [CMxT, BDiff]
    
    # number of cells
    nCells = [nCX,nCCY,1]

    # grading
    grading = [grX,grYC,grZ]

    # create the block
    CDiff = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    # define special edges
    e32 = list()
    e76 = list()

    for point in diffuserPnts:
        e32.append(((xE-xC)*point[0]+xC, (yE02-yE01)*point[1]+yE01, zC-((yE02-yE01)*point[1]+yE01)/yMax*dZ))
        e76.append(((xE-xC)*point[0]+xC, (yE02-yE01)*point[1]+yE01, zC+((yE02-yE01)*point[1]+yE01)/yMax*dZ))

    fvMesh.addEdge("polyLine", CDiff.retEYEZ0(), e32)
    fvMesh.addEdge("polyLine", CDiff.retEYEZE(), e76)
    
    #-----------------------------------------------------------------------   
    # - post-diff tube part one
    # APost
    AC,BC,CC = A4,B4,C4
    AN,BN,CN = A4,B4,C4
    
    xC,yC,zC = xE,y0,z0
    xE,yE01,yE02 = xC+LPost,yC+AC,yC+AN

    # vertex coordinates
    vertices = [
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC,zC-yC/yMax*dZ],
            [xE,yC,zC-yC/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [ADiff]
    
    # number of cells
    nCX = int(round((xE-xC)/dX*0.9))
    nCells = [nCX,nCAY,1]

    # grading
    grading = [grXPost,grY,grZ]

    # create the block
    APost = fvMesh.addBlock(vertices, neighbors, nCells, grading)
    
    # BPost
    xC,yC01,yC02,zC = xC,yE01,yE02,zC
    xE,yE01,yE02 = xE,yC01+BC,yC02+BN

    # vertex coordinates
    vertices = [
            [xC,yC01,zC-yC01/yMax*dZ],
            [xE,yC02,zC-yC02/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC01,zC+yC01/yMax*dZ],
            [xE,yC02,zC+yC02/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [BDiff, APost]
    
    # number of cells
    nCells = [nCX,nCBY,1]

    # grading
    grading = [grXPost,grYB,grZ]

    # create the block
    BPost = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    # CPost
    xC,yC01,yC02,zC = xC,yE01,yE02,zC
    xE,yE01,yE02 = xE,yC01+CC,yC02+CN

    # vertex coordinates
    vertices = [
            [xC,yC01,zC-yC01/yMax*dZ],
            [xE,yC02,zC-yC02/yMax*dZ],
            [xE,yE02,zC-yE02/yMax*dZ],
            [xC,yE01,zC-yE01/yMax*dZ],
            [xC,yC01,zC+yC01/yMax*dZ],
            [xE,yC02,zC+yC02/yMax*dZ],
            [xE,yE02,zC+yE02/yMax*dZ],
            [xC,yE01,zC+yE01/yMax*dZ],
        ]

    # neighboring blocks
    neighbors = [CDiff, BPost]
    
    # number of cells
    nCells = [nCX,nCCY,1]

    # grading
    grading = [grXPost,grYC,grZ]

    # create the block
    CPost = fvMesh.addBlock(vertices, neighbors, nCells, grading)

    #-----------------------------------------------------------------------
    # prepare boundaries

    # -- symmetries - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # - symmetry plane boundaries
    wedgeZ0 = []
    for block in fvMesh.blocks:
        wedgeZ0.append(block.retFXY0())
    
    fvMesh.addPatch("wedgeZ0", "wedge", wedgeZ0)
    
    wedgeZE = []
    for block in fvMesh.blocks:
        wedgeZE.append(block.retFXYE())
    
    fvMesh.addPatch("wedgeZE", "wedge", wedgeZE)
    
    # -- walls - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # - wall boundary
    # -- walls for refinement
    # - upper walls
    upperWalls = []
    upperWalls.append(suctionChamber.retFXZE())
    upperWalls.append(CGap.retFXZE())
    upperWalls.append(CConv.retFXZE())
    upperWalls.append(CMxT.retFXZE())
    upperWalls.append(CDiff.retFXZE())
    upperWalls.append(CPost.retFXZE())
    
    fvMesh.addPatch("upperWalls", "wall", upperWalls)

    # - other walls
    otherWalls = []
    otherWalls.append(nozzleBody.retFXZE())
    otherWalls.append(nozzleConv.retFXZE())
    otherWalls.append(nozzleTube.retFXZE())
    otherWalls.append(BGap.retFYZ0())
    otherWalls.append(suctionChamber.retFXZ0())
    
    fvMesh.addPatch("otherWalls", "wall", otherWalls)

    # bottom walls
    bottomWalls = []
    bottomWalls.append(nozzleBody.retFXZ0())
    bottomWalls.append(nozzleConv.retFXZ0())
    bottomWalls.append(nozzleTube.retFXZ0())
    bottomWalls.append(AGap.retFXZ0())
    bottomWalls.append(AConv.retFXZ0())
    bottomWalls.append(AMxT.retFXZ0())
    bottomWalls.append(ADiff.retFXZ0())
    bottomWalls.append(APost.retFXZ0())
    
    fvMesh.addPatch("bottomWalls", "wall", bottomWalls)

    # -- in/outs - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # - inlet boundary
    inlet = []
    inlet.append(nozzleBody.retFYZ0())
    
    fvMesh.addPatch("inlet", "patch", inlet)
    
    # - outlet boundary
    outlet = []
    outlet.append(APost.retFYZE())
    outlet.append(BPost.retFYZE())
    outlet.append(CPost.retFYZE())
    
    fvMesh.addPatch("outlet", "patch", outlet)
    
    # - suction boundary
    suction = []
    suction.append(suctionChamber.retFYZ0())

    fvMesh.addPatch("suction", "patch", suction)
    
    #-------------------------------------------------------------------
    # FILE GENERATION
    #-------------------------------------------------------------------
    fvMesh.writeBMD(caseDir + "system")

    return [blockMeshClass, badMesh]
