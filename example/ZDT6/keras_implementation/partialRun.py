import sys

sys.path.insert(1, "../../../src")
sys.path.insert(1, "../../../thirdParty")
from CFDNNetAdaptV4 import CFDNNetAdapt
import evalFunctions as evalF


if __name__ == "__main__":
    # declare CFDNNetAdapt
    algorithm = CFDNNetAdapt(evalF.dnnEvaluation, evalF.smpEvaluation, drawTrainingPlot=True, saveTrainingHistory=True, minN=8,
                             maxN=16, nSeeds=1, iMax=1, activationFunction="tanh", lm_optimizer=True, validationFreq=1,
                             nValFails=10, smpDir=r"../00_prepData/")

    # run
    algorithm.run()
