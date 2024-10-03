from CFDNNetAdaptV4 import CFDNNetAdapt


if __name__ == "__main__":
    # declare CFDNNetAdapt
    algorithm = CFDNNetAdapt(drawTrainingPlot=True, saveTrainingHistory=True, minN=16, maxN=16, nSeeds=3, iMax=3,
                             activationFunction="tanh")

    # run
    algorithm.run()
