import sys

sys.path.insert(1, "../../src")
sys.path.insert(1, "../../thirdParty")
from CFDNNetAdaptV4 import CFDNNetAdapt
import evalFunctions as evalF
from postProcess import postProcess


if __name__ == "__main__":
    n_params = 10
    n_outputs = 2

    # declare CFDNNetAdapt
    algorithm = CFDNNetAdapt(
        dnnEvaluation=evalF.dnnEvaluationZDT,
        smpEvaluation=evalF.smpEvaluationZDT3,
        # lm_optimizer=True,
        activationFunction="silu",
        nPars=n_params,
        nObjs=n_outputs,
        nOuts=n_outputs,
        mainDir="01_algoRuns/",
        smpDir="00_prepData/",
        prbDir="ZDT4/",
        dataNm="10_platypusAllSolutions.dat",
        nSam=2000,
        deltaNSams=[2000],
        nNN=1,
        minN=14,
        maxN=32,
        nHidLay=3,
        tol=5e-2,
        iMax=3,
        dRN=0,
        nComps=0,  # smpEvaluation not used
        nSeeds=1,
        trainPro=70,
        valPro=30,
        testPro=0,
        kMax=15000,
        rEStop=1e-6,
        verbose=True,
        pMin=0.0,
        pMax=1.0,
        offSize=500,
        popSize=500,
        nGens=60,
        drawTrainingPlot=True,
        saveTrainingHistory=True,
        patience=300,
        validationFreq=1,
        dropout=0.15,
        # batchNorm=True,
        netStructs=[[n_params, 64, 32, n_outputs]],
        batch_size=256,
    )

    algorithm.run()

    runDir = algorithm.getRunDir()
    postProcess(runDir=runDir, xName="f1", yName="f2")
    print("Post processing done.")