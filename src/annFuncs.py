
import math
import numpy as np
from scipy import interpolate 

def bSpline(controlPnts, N):
    x = []
    y = []
    for point in controlPnts:
        x.append(point[0])
        y.append(point[1])
    
    xx = np.linspace(min(x), max(x), N)
    
    k = len(x)-1
    if k>5:
        k = 5
    
    t, c, k = interpolate.splrep(x, y, s=0, k=k)
    yy = interpolate.splev(xx, (t,c,k))
    
    edgePnts = []
    for i in range(len(xx)-1):
        point = [xx[i], yy[i]]
        edgePnts.append(point)
    
    return edgePnts

def roughBSpline(controlPnts, N):
    edgePnts = bSpline(controlPnts, N)

    roughPnts = [edgePnts[0]]
    every = 1
    for i in range(1,len(edgePnts)-1):
        if i%every == 0:
            roughPnts.append(edgePnts[i])

    roughPnts.append(edgePnts[-1])

    allPnts = polyLine(roughPnts, N)

    return allPnts

def polyLine(controlPnts, N):
    x = np.linspace(0,1,N)
    y = list()

    for i in range(len(x)):
        for j in range(len(controlPnts)-1):
            left = controlPnts[j]
            right = controlPnts[j+1]

            if x[i] >= left[0] and x[i] < right[0]:
                toAppend = (right[1]-left[1])/(right[0]-left[0])*(x[i]-left[0])+left[1]
                y.append(toAppend)

    y.append(controlPnts[-1][1])
    y = np.array(y)

    controlPnts = np.array([x,y]).T

    return controlPnts

def sine(depth, nWaves, N):
    x = np.linspace(0,2*math.pi,N)
    y = [-1*(depth*math.sin(math.sin(math.sin(math.sin(nWaves/2*i))))**2/math.sin(math.sin(math.sin(1)))**2) for i in x]

    sinPnts = np.array([x/2/math.pi,y]).T

    return sinPnts

def arcOf3Pnts(pnts, N):
    x1,y1 = pnts[0]
    x2,y2 = pnts[1]
    x3,y3 = pnts[2]

    xC = (((x1**2-x2**2+y1**2-y2**2)*(y2-y3))-((x2**2-x3**2+y2**2-y3**2)*(y1-y2)))/(2*(((x1-x2)*(y2-y3))-((x2-x3)*(y1-y2))))
    yC = (x1**2-x2**2-2*xC*(x1-x2)+y1**2-y2**2)/(2*(y1-y2))
    r2 = (x1-xC)**2+(y1-yC)**2

    xx = np.linspace(x1,x3,N)
    yy = list()

    for x in xx:
        y = np.sign(yC)*math.sqrt(r2-(x-xC)**2)+yC
        if not (y >= y1 and y <= y3):
            y = yC-(y-yC)

        yy.append(y)

    arcPnts = np.array([xx,yy]).T

    return arcPnts
