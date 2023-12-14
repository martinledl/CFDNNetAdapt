# CFDNNetAdapt
The CFDNNetAdapt is a hybrid CFD-DNN optimization algorithm for CFD-based shape optimization. The algorithm combines CFD with multi-objective multi-parameter optimization performed via MOEA with directly incorporated DNNs. The DNN architecture is searched for automatically and accelerate the mid-to-late MOEA iterations.

## Third party code
MOEA --  D. Hadka, Platypus, A Free and Open Source Python Library for Multiobjective Optimization, 2020. URL: https://github.com/Project-Platypus/Platypus

DNNs --  D. Atabay, Institute for Energy Economy and Application Technology,665 Technische Universität München, pyrenn: A recurrent neural network tool-box for python and matlab, 2018. URL: https://pyrenn.readthedocs.io/en/latest/.

## Cite this work as (article prepared for submission)
L. Kubíčková and O. Gebouský and J. Haidl and M. Isoz.: CFDNNetAdapt: An adaptive shape optimization algorithm coupling CFD and Deep Neural Networks. Submitted in December 2023.

## Compatability
Prepared for python3 (https://www.python.org/downloads/release/python-31010/) and OpenFOAMv8 (https://openfoam.org/version/8/).

## Prepapare the python environment
used python3 packages -- os, io, math, sys, shutil, numpy, scipy, re, copy, csv, dill, multiprocessing, glob, subprocess, operator, random, datetime
python3 packages used by thirdParty codes -- six, pandas, functools, traceback, mpi4py, unittest, pickle, abc, time, logging, collections, sets

## Example run
cd ./example/convDifShapeOptim && python3 Allrun.py

## Testing run
cd ./example/convDifShapeOptim && python3 testRun.py

## License
CFDNNetAdapt is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software  Foundation, either version 3 of the License, or (at your option) any later version.  See http://www.gnu.org/licenses/, for a description of the GNU General Public License terms under which you can copy the files.
