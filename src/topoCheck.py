import numpy as np


def topoCheck(data, correct=False, yLim=0):
    yDim, xDim = np.shape(data)
    if correct:
        correctedData = np.copy(data)

    totalMess1 = 0

    # check row from top to bottom
    rowCheckTB = [False] * yDim

    rowChecks = list()
    check = [True] * xDim
    for i in range(yDim):
        newCheck = [False] * xDim
        for j in range(xDim):
            if data[i][j] == True and check[j] == True:
                newCheck[j] = True

        for j in range(1, xDim):
            if data[i][j] == True and newCheck[j - 1] == True:
                newCheck[j] = True

        for j in range(xDim - 2, -1, -1):
            if data[i][j] == True and newCheck[j + 1] == True:
                newCheck[j] = True

        newCheck = np.array(newCheck)
        if np.all(data[i] == newCheck):
            rowCheckTB[i] = True

        else:
            if correct:
                for j in range(xDim):
                    if data[i][j] == True and newCheck[j] == False:
                        correctedData[i][j] = False

            totalMess1 += sum(data[i]) - sum(newCheck)
            rowCheckTB[i] = False

        check = newCheck
        rowChecks.append(check)

    if correct:
        data = np.copy(correctedData)

    # check columns from left to right
    totalMess0 = 0
    flowAccessLR = 0
    colCheckLR = [False] * xDim

    check = list()
    for j in range(yDim):
        if j < yLim:
            check.append(True)

        else:
            check.append(False)

    for i in range(xDim):
        newCheck = [True] * yDim
        for j in range(yDim):
            if data[j][i] == False and check[j] == False:
                newCheck[j] = False
                flowAccessLR = i + 1

        for j in range(1, yDim):
            if data[j][i] == False and newCheck[j - 1] == False:
                newCheck[j] = False
                flowAccessLR = i + 1

        for j in range(yDim - 2, -1, -1):
            if data[j][i] == False and newCheck[j + 1] == False:
                newCheck[j] = False
                flowAccessLR = i + 1

        newCheck = np.array(newCheck)
        if np.all(data[:, i] == newCheck):
            colCheckLR[i] = True

        else:
            if correct:
                for j in range(yDim):
                    if data[j][i] == False and newCheck[j] == True:
                        correctedData[j][i] = True

            totalMess0 += sum(newCheck) - sum(data[:, i])
            colCheckLR[i] = False

        check = newCheck

    # check columns from right to left
    flowAccessRL = 0
    colCheckRL = [False] * xDim

    # ~ check = [False]*yDim
    for i in range(xDim - 1, -1, -1):
        newCheck = [True] * yDim
        for j in range(yDim):
            if data[j][i] == False and check[j] == False:
                newCheck[j] = False
                flowAccessRL = xDim - i

        for j in range(1, yDim):
            if data[j][i] == False and newCheck[j - 1] == False:
                newCheck[j] = False
                flowAccessRL = xDim - i

        for j in range(yDim - 2, -1, -1):
            if data[j][i] == False and newCheck[j + 1] == False:
                newCheck[j] = False
                flowAccessRL = xDim - i

        newCheck = np.array(newCheck)
        if np.all(data[:, i] == newCheck):
            colCheckRL[i] = True

            if correct:
                if colCheckLR[i] == False:
                    for j in range(yDim):
                        if correctedData[j][i] == True and newCheck[j] == False:
                            correctedData[j][i] = False

        else:
            if correct:
                for j in range(yDim):
                    if colCheckLR[i] == False and data[j][i] == False:
                        if correctedData[j][i] == True:
                            correctedData[j][i] = newCheck[j]

                    if colCheckLR[i] == False and correctedData[j][i] == True and newCheck[j] == False:
                        correctedData[j][i] = False

            totalMess0 += sum(newCheck) - sum(data[:, i])
            colCheckRL[i] = False

        check = newCheck

    # sum up the column checks
    totalColCheck = [False] * xDim
    for i in range(xDim):
        if colCheckLR[i] == False and not colCheckRL[i] == False:
            if True in colCheckLR[i:]:
                totalColCheck[i] = True

        if colCheckRL[i] == False and not colCheckLR[i] == False:
            if True in colCheckRL[:i]:
                totalColCheck[i] = True

        if colCheckLR[i] == True and colCheckRL[i] == True:
            totalColCheck[i] = True

    flowAccess = max(flowAccessLR, flowAccessRL)

    if not False in rowCheckTB and not False in totalColCheck:
        if correct:
            return [flowAccess, correctedData]

        else:
            return [True, 0, 0, 0]

    else:
        if correct:
            return [flowAccess, correctedData]

        else:
            return [False, totalMess1, totalMess0, flowAccess]
