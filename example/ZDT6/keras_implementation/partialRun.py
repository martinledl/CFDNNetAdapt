import sys

sys.path.insert(1, "../../../src")
sys.path.insert(1, "../../../thirdParty")
from CFDNNetAdaptV4 import CFDNNetAdapt
import evalFunctions as evalF
from postProcess import postProcess


if __name__ == "__main__":
    # declare CFDNNetAdapt
    # algorithm = CFDNNetAdapt(evalF.dnnEvaluation, evalF.smpEvaluation, drawTrainingPlot=True, saveTrainingHistory=True, minN=8,
    #                          maxN=16, nSeeds=1, iMax=1, activationFunction="tanh", lm_optimizer=True, validationFreq=1,
    #                          nValFails=10, smpDir=r"../00_prepData/")

    algorithm = CFDNNetAdapt(
        dnnEvaluation=evalF.dnnEvaluation,
        smpEvaluation=evalF.smpEvaluation2,
        lm_optimizer=False,
        activationFunction="tanh",
        nPars=2,
        nObjs=2,
        nOuts=2,
        mainDir="01_algoRuns/",
        smpDir="../00_prepData/",
        nSam=4000,
        deltaNSams=[2000],
        nNN=1,
        minN=32,
        maxN=64,
        nHidLay=2,
        tol=5e-2,
        iMax=1,
        nComps=0,  # smpEvaluation not used
        nSeeds=1,
        trainPro=70,
        valPro=30,
        testPro=0,
        kMax=10000,
        rEStop=1e-5,
        verbose=True,
        drawTrainingPlot=True,
        saveTrainingHistory=False,
        patience=100,
        validationFreq=1,
        dropout=0.15,
        # batchNorm=True,
        # netStructs=[[2, 64, 32, 2]],
        batch_size=256,
    )

    algorithm.run()

    runDir = algorithm.getRunDir()
    postProcess(runDir=runDir, xName="f1", yName="f2")
    print("Post processing done.")
