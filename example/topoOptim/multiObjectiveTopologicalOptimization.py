import os
import sys
import csv
import datetime
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, InputLayer, Lambda, ReLU, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanAbsolutePercentageError, Huber, MeanSquaredError

sys.path.append('../../thirdParty')
import platypusModV3 as platypus

sys.path.append('../../src')
from evalFunctions import auxFunction
from postProcess import plot_data
from compare_matrices import hamming_similarity, jaccard_similarity, dice_similarity, earth_mover_distance


# create a model
def getModel(data_shape):
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 5), strides=1, activation=None, padding='same', input_shape=data_shape))
    # model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 5), padding='same'))

    model.add(Conv2D(32, kernel_size=(3, 5), strides=1, activation=None, padding='same'))
    # model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 5), padding='same'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))
    return model


def trainModel(data_shape):
    smpDir = './00_prepTopoOptimData/'
    prbDir = ""
    dataNm = "feasible52x10-capped10.csv"
    dataPath = smpDir + prbDir + dataNm

    df = pd.read_csv(dataPath)
    df.drop_duplicates(inplace=True)

    # train model
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    X, y = df.drop(['pressureRecoveryFactor', 'uniformityIndex'], axis=1), df[
        ['pressureRecoveryFactor', 'uniformityIndex']]

    model = getModel(data_shape)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[RootMeanSquaredError()])
    model.summary(line_length=120)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    X_train = np.array(X_train).reshape(-1, data_shape[0], data_shape[1], data_shape[2])
    X_val = np.array(X_val).reshape(-1, data_shape[0], data_shape[1], data_shape[2])

    history = model.fit(X_train, y_train, epochs=300, batch_size=256, callbacks=[early_stopping],
                        validation_data=(X_val, y_val))

    return model, history


def run_optimization(dataDir, injectedPopFile, xDim, yDim, yLim, logFile, outFile, model):
    # optimization settings
    method = "NSAGII"  # optimization algorithm
    nGens = 20  # number of generations
    parallelNum = 8  # number of parallel running processes
    popSize = parallelNum * 200  # size of a population
    nIter = popSize * nGens  # number of function evaluations

    nPars = 1  # number of parameters
    nObjs = 2  # number of objectives

    gamma0 = 5.0  # for penalty function
    gamma1 = 10.0  # for penalty function
    gammaF = 100.0  # for penatly function

    listToPrint = ["xDim", "yDim", "yLim", "method", "nGens", "popSize", "nIter", "gamma0", "gamma1", "gammaF"]

    with open(logFile, 'a') as file:
        for var in listToPrint:
            file.write(var + " = " + str(eval(var)) + "\n")

    with open(outFile, 'w') as file:
        file.write(f'caseID,good{",bit" * (xDim * yDim)},pressureRecoveryFactor,uniformityIndex\n')

    problem = platypus.Problem(nPars, nObjs)
    problem.types[:] = [platypus.Binary(xDim * yDim)]
    problem.function = auxFunction
    problem.kwargs = {'net': model, 'yDim': yDim, 'xDim': xDim, 'yLim': yLim, 'gamma0': gamma0, 'gamma1': gamma1,
                      'gammaF': gammaF, 'outFile': outFile, 'nObjs': nObjs}

    # load injectedPopulation
    if injectedPopFile is not None:
        injectedSolutions = list()

        with open(injectedPopFile, 'r') as file:
            reader = csv.reader(file)

            for line in reader:
                individuum = platypus.core.Solution(problem)
                individuum.variables = list()
                for i in line:
                    if i == "True":
                        toAppend = True
                    else:
                        toAppend = False

                    individuum.variables.append(toAppend)

                individuum.variables = [individuum.variables]
                individuum.objectives = [0.0] * nObjs
                individuum.evaluated = False
                injectedSolutions.append(individuum)

        # run the optimization algorithm
        with platypus.MultiprocessingEvaluator(parallelNum) as evaluator:
            algorithm = platypus.NSGAII(problem, population_size=popSize,
                                        generator=platypus.InjectedPopulation(injectedSolutions), evaluator=evaluator,
                                        variator=platypus.GAOperatorWithTopoCorrection(platypus.HUX(),
                                                                                       platypus.BitFlip(),
                                                                                       xDim,
                                                                                       yDim, yLim))
            algorithm.run(nIter)

        with open(dataDir + "optimOut.plat", 'wb') as file:
            pickle.dump(
                [algorithm.population, algorithm.result, method, problem],
                file,
                protocol=2
            )
    else:
        with open(logFile, 'a') as file:
            file.write("No injectedPopFile, quitting\n")


def compare_with_best(result, logFile):
    sorted_result = sorted(result, key=lambda x: x.objectives[0] + x.objectives[1])
    A = np.array(sorted_result[0].variables[0]).reshape((10, 52))

    df = pd.read_csv('00_prepTopoOptimData/feasible52x10.csv')
    sorted_df = df.sort_values(by=["pressureRecoveryFactor", "uniformityIndex"])
    B = np.array(sorted_df.iloc[0, 2:]).reshape((10, 52))

    output = "Hamming Similarity: " + str(hamming_similarity(A, B)) + "\n"
    output += "Jaccard Similarity: " + str(jaccard_similarity(A, B)) + "\n"
    output += "Dice Similarity: " + str(dice_similarity(A, B)) + "\n"
    output += "Earth Mover's Distance: " + str(earth_mover_distance(A, B)) + "\n"

    with open(logFile, 'a') as file:
        file.write(output)

    print(output)


def main():
    # universal arguments
    dataDir = os.getcwd() + "/ZZ_dataDirs/topoOptim_" + str(datetime.datetime.now().strftime("%d%m%Y%H%M%S")) + "/"
    injectedPopFile = "./00_prepTopoOptimData/initialPopulation.dat"
    os.makedirs(dataDir)

    # discretization
    xDim = 52
    yDim = 10
    yLim = 3

    # train model
    data_shape = (yDim, xDim, 1)
    model, history = trainModel(data_shape)
    model.save(dataDir + "model.keras")

    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(dataDir + "history.png")

    # log
    name = "topologicalOptimizationMxTLDiff"
    logFile = dataDir + name + ".log"
    outFile = dataDir + "allSolutions.dat"

    with open(logFile, 'a') as file:
        file.write("\nstartTime = " + str(datetime.datetime.now().strftime("%d/%m/%Y %X")) + "\n")
        file.write("===================SET UP=====================\n")
        file.write("\n")
        model.summary(print_fn=lambda x: file.write(x + '\n'), line_length=120)

    with open(outFile, 'a') as file:
        file.write("caseID,note," + "bit," * (xDim * yDim) + "pressureRecoveryFactor,uniformityIndex\n")
        file.flush()
        file.close()

    # optimalization itself
    run_optimization(dataDir, injectedPopFile, xDim, yDim, yLim, logFile, outFile, model)

    with open(f'{dataDir}optimOut.plat', 'rb') as file:
        [population, result, name, problem] = pickle.load(file, encoding="latin1")

    plot_data(None, result, f'{dataDir}/result.png', limit=None)

    compare_with_best(result, logFile)

    # log
    with open(logFile, 'a') as file:
        file.write("==============================================\n")
        file.write("endTime = " + str(datetime.datetime.now().strftime("%d/%m/%Y %X")) + "\n")


if __name__ == '__main__':
    main()

    # experimentDir = './ZZ_dataDirs/topoOptim_08112024134501/'
    #
    # with open(f'{experimentDir}optimOut.plat', 'rb') as file:
    #     [population, result, name, problem] = pickle.load(file, encoding="latin1")
    #
    # plot_data(None, result, f'{experimentDir}/allData.png', limit=100)