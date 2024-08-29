
### definition of testing functions
# included functions:
#   ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

import math
import numpy as np
from scipy.optimize import root_scalar

## ZDT1
def ZDT1(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9*sum(x[1:])/(n - 1)
    f2 = g*(1 - math.sqrt(x[0]/g))

    return [f1, f2]

def optSolsZDT1(n, nPars):
    x1s = np.linspace(0.0, 1.0, n)
    optSols = list()
    for i in range(n):
        pars = [x1s[i]] + [0]*(nPars - 1)
        optSols.append(pars)

    return optSols

## ZDT2
def ZDT2(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9*sum(x[1:])/(n - 1)
    f2 = g*(1 - (x[0]/g)**2)

    return [f1, f2]

def optSolsZDT2(n, nPars):
    x1s = np.linspace(0.0, 1.0, n)
    optSols = list()
    for i in range(n):
        pars = [x1s[i]] + [0]*(nPars - 1)
        optSols.append(pars)

    return optSols

# ZDT3
def ZDT3(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9*sum(x[1:])/(n - 1)
    f2 = g*(1 - math.sqrt(x[0]/g) - x[0]/g*math.sin(10*math.pi*x[0]))

    return [f1, f2]

# alias optimal curve
def optCurveZDT3(x0):
    f = 1 - math.sqrt(x0) - x0*math.sin(10*math.pi*x0)
    return f

# optimal curve derivative
def deroptZDT3(x0):
    diff = -1/(2*math.sqrt(x0)) - math.sin(10*math.pi*x0) - 10*math.pi*x0*math.cos(10*math.pi*x0)
    return diff

# real optimal solutions
def optSolsZDT3(n, nPars):
    x1s = np.linspace(0.0, 1.0, n)
    badSols = list()
    for i in range(n):
        pars = [x1s[i]] + [0]*(nPars - 1)
        badSols.append(pars)
    
    badf1s = list()
    badf2s = list()
    for i in range(len(badSols)):
        f1, f2 = ZDT3(badSols[i])
        badf1s.append(f1)
        badf2s.append(f2)
    
    fs = [optCurveZDT3(i) for i in x1s]
    
    # find roots
    mins = [0.07, 0.24, 0.44, 0.64, 0.84]
    maxs = [ 0.1, 0.27, 0.46, 0.66, 0.86]
    sols = [0.0]
    for i in range(len(mins)):
        solution = root_scalar(deroptZDT3, x0 = mins[i], x1 = maxs[i])
        sols.append(solution.root)
    sols.append(1.0)
    fds = [optCurveZDT3(i) for i in sols]
    
    # filter optimal solutions
    optSols = list()
    for i in range(n):
        x1 = x1s[i]
        f2 = badf2s[i]
        for j in range(1,len(sols)):
            left = sols[j-1]
            right = sols[j]
            top = fds[j-1]
            bottom = fds[j]
    
            if x1 >= left and x1 < right:
                if f2 <= top and f2 > bottom:
                    optSols.append(badSols[i])
                else:
                    continue

    return optSols

## ZDT4
def ZDT4(x):
    n = len(x)
    f1 = x[0]
    suma = 0
    for i in range(1,n):
        suma += x[i]**2 - 10*math.cos(4*math.pi*x[i])
    g = 1 + 10*(n - 1) + suma
    f2 = g*(1 - math.sqrt(x[0]/g))

    return [f1, f2]

def optSolsZDT4(n, nPars):
    x1s = np.linspace(0.0, 1.0, n)
    optSols = list()
    for i in range(n):
        pars = [x1s[i]] + [0]*(nPars - 1)
        optSols.append(pars)

    return optSols

## ZDT6
def ZDT6(x):
    n = len(x)
    f1 = 1 - math.exp(-4*x[0])*(math.sin(6*math.pi*x[0]))**6
    g = 1 + 9*(sum(x[1:])/(n - 1))**0.25
    f2 = g*(1 - (f1/g)**2)

    return [f1, f2]

def optSolsZDT6(n, nPars):
    x1s = np.linspace(0.0, 1.0, n)
    optSols = list()
    for i in range(n):
        pars = [x1s[i]] + [0]*(nPars - 1)
        optSols.append(pars)

    return optSols
