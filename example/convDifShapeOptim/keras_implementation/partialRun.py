import sys

sys.path.insert(1, "../../../src")
sys.path.insert(1, "../../../thirdParty")
from CFDNNetAdaptV4 import CFDNNetAdapt
import evalFunctions as evalF


if __name__ == "__main__":
    # declare CFDNNetAdapt
    algorithm = CFDNNetAdapt(
        dnnEvaluation=evalF.dnnEvaluation2,
        smpEvaluation=evalF.smpEvaluation2,
        lm_optimizer=True,
        activationFunction="silu",
        nPars=14,
        nObjs=2,
        nOuts=1,
        mainDir="01_algoRuns/",
        smpDir="../00_prepCFDData/",
        prbDir="14_LConv2CPLDiff4CP/",
        dataNm="10_platypusCFDAllSolutions.dat",
        minMax="12_minMaxAng.dat",
        nSam=2000,
        deltaNSams=[2000],
        nNN=1,
        minN=5,
        maxN=20,
        nHidLay=3,
        tol=5e-2,
        iMax=4,
        dRN=0,
        nComps=0,  # smpEvaluation not used
        nSeeds=1,
        trainPro=75,
        valPro=15,
        testPro=10,
        kMax=10000,
        rEStop=1e-5,
        verbose=True,
        pMin=0.0,
        pMax=1.0,
        offSize=500,
        popSize=500,
        nGens=30,
        drawTrainingPlot=True,
        saveTrainingHistory=True,
        nValFails=15,
    )

    algorithm.run()
