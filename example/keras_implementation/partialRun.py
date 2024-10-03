from CFDNNetAdaptV4 import CFDNNetAdapt


if __name__ == "__main__":
    # declare CFDNNetAdapt
    algorithm = CFDNNetAdapt(drawTrainingPlot=True, minN=16, maxN=16, nNN=1, iMax=1)

    # run
    algorithm.run()
