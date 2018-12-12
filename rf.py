"""
 _____           _              _____         _
| __  |___ ___ _| |___ _____   |   __|___ ___| |_ _ _ ___ ___ ___
|    -| .'|   | . | . |     |  |   __| -_| .'|  _| | |  _| -_|_ -|
|__|__|__,|_|_|___|___|_|_|_|  |__|  |___|__,|_| |___|_| |___|___|

Fast Random Kernelized Features: Extending Support Vector Machine
Classification for High Dimensional IDC Classification

Authors
-------
Tianshu Huang <tianshu.huang@utexas.edu>
  | Department of Electrical and Computer Engineering
  | University of Texas at Austin
Nimit Kalra   <nimit@utexas.edu>
  | Department of Computer Science
  | University of Texas at Austin

Dependencies
------------
* numpy
* scikit-learn
* scikit-image
* print (https://github.com/thetianshuhuang/print)
* syllabus (https://github.com/thetianshuhuang/syllabus)

Usage
-----
Run the module with python rf.py arg1=value arg2=value etc.

Parameter  Default  Description
-----------------------------------------------------------
ptrain     0.01     Percent of training images to use
ptest      0.1      Percent of test images to use
fdim       5000     Feature space dimensionality
knn        False    Use Color K Means?
k          5        Number of means to use
kernel     G        Kernel type ("G", "L", or "C")
ntrain     -25      Number of patients used for training
ntest      25       Number of patients used for testing
cores      8        Number of cores (processes) to use
ftype      F        type of feature to use ("F" or "B")

References
----------
[1] Ali Rahimi and Benjamin Recht. "Random Features for Large-Scale Kernel
    Machines." Advances in Neural Information Processing Systems. 2008.
[2] Wu, Lingfei, et al. "Revisiting Random Binning Features: Fast Convergence
    and Strong Parallelizability." Proceedings of the 22nd ACM SIGKDD
    International Conference on Knowledge Discovery and Data Mining. ACM,
    2016.
[3] Netzer, Yuval, et al. "Reading Digits in Natural Images with Unsupervised
    Feature Learning." NIPS Workshop on Deep Learning and Unsupervised Feature
    Learning. Vol. 2011. No. 2. 2011.
[4] Janowczyk, Andrew. "Invasive Ductal Carcinoma Identification Dataset."
    http://www.andrewjanowczyk.com/deep-learning/.
[5] Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
    and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
    and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
    Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.
    "Scikit-learn: Machine Learning in Python." Journal of Machine Learning
    Research. JMLR, 2011.
"""

import sys
from print import *
from syllabus import Task

from randomfeatures import CKM
from config import RF_PARAMS
from helpers import train, make_feature, make_datasets
from tester import IDCDataset, PATIENTS


if __name__ == "__main__":

    # Set log file
    putil.LOG_FILE = 'results_' + '_'.join(sys.argv[1:]) + '.txt'

    # Print out header
    div.div('- -', BOLD, label='Random Feature Support Vector Classifier')

    # Hyperparameters
    args = RF_PARAMS.parse()

    # Print out arguments
    print('\n')
    args.print()
    print('\n\n')

    # Run program
    div.div('-', BR + BLUE, label='Main Program')
    main = Task(name='RF', desc="Random Feature Support Vector Classifier")

    # Generate color map if flag set
    if args.get('knn'):
        ckm = CKM(args.get('k'), IDCDataset(PATIENTS, p=0.01))
        idim = args.get('k')
    else:
        idim = 7500

    # Make random feature
    rf = make_feature(idim=idim, task=main,
                      **args.subdict('ftype', 'kernel', 'fdim', 'cores'))

    # Load datasets
    [dataset,
     test_dataset,
     tester,
     debugtester] = make_datasets(
        rf.transform, **args.subdict('cores', 'ntrain', 'ptrain', 'ptest'),
        feature=ckm.map if args.get('knn') else None, main=main)

    # Train model
    rfsvm = train(dataset, main.subtask())

    # Tester
    tester.loss(rfsvm, task=main)

    # Debug tester
    debugtester.loss(rfsvm, task=main)
    main.done("Program finished.")
