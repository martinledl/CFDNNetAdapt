from CFDNNetAdaptV4 import CFDNNetAdapt


if __name__ == "__main__":
    # declare CFDNNetAdapt
    algorithm = CFDNNetAdapt(drawTrainingPlot=True, saveTrainingHistory=True, minN=8, maxN=16, nSeeds=1, iMax=1,
                             activationFunction="tanh", lm_optimizer=True, validationFreq=1)

    # run
    algorithm.run()
