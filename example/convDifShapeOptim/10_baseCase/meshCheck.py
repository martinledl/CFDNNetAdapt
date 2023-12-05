
import sys
import numpy as np

def __main__():
    checks = list()
    
    fileName = 'log.checkMesh'
    with open(fileName, 'r') as file:
        line = file.readline()
    
        while line:
            if 'Checking faces in error :' in line:
                line = file.readline()
                while not line == '\n' and not "<<" in line:
                    check = int(line.split(': ')[-1])
                    checks.append(check)
                    line = file.readline()
    
            line = file.readline()
    
    checks = np.array(checks)
    
    if np.all(checks == 0):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    __main__()
