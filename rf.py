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
Run the module with python rf.py arg1=value arg2=value etc., where parameters
are passed as keyword arguments joined by '=' and separated by spaces.

Parameter  Default  Description
-----------------------------------------------------------
ptrain     0.01     Percent of training images to use
ptest      0.1      Percent of test images to use
fdim       5000     Feature space dimensionality
knn        False    Use color k-means?
k          5        Number of means to use
kernel     G        Kernel type ("G", "L", or "C")
ntrain     -25      Number of patients used for training
ntest      25       Number of patients used for testing
cores      8        Number of cores (processes) to use
ftype      F        Type of feature to use ("F" or "B")

References
----------
[1] Ali Rahimi and Benjamin Recht. "Random Features for Large-Scale Kernel
    Machines." Advances in Neural Information Processing Systems. 2008.
[2] Wu, Lingfei, et al. "Revisiting Random Binning Features: Fast Convergence
    and Strong Parallelizability." Proceedings of the 22nd ACM SIGKDD
    International Conference on Knowledge Discovery and Data Mining. ACM,
    2016.
[3] Janowczyk, Andrew. "Invasive Ductal Carcinoma Identification Dataset."
    http://www.andrewjanowczyk.com/deep-learning/.
[4] Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
    and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
    and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
    Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.
    "Scikit-learn: Machine Learning in Python." Journal of Machine Learning
    Research. JMLR, 2011.
[5] van der Walt, Stefan, Schonberger, Johannes L., Nunez-Iglesias, Juan,
    Boulogne, Franc¸ois, Warner, Joshua D., Yager, Neil, Gouillart, Emmanuelle,
    Yu, Tony, and the scikit-image contributors. scikit-image: image processing
    in Python. PeerJ, 2:e453, 6 2014. ISSN 2167-8359. doi: 10.7717/peerj.453.
    URL http://dx.doi.org/10.7717/peerj.453.
"""

import sys
from print import *
from syllabus import Task

from randomfeatures import CKM
from config import RF_PARAMS
from helpers import train, make_feature, make_trainset, make_testset
from tester import IDCDataset, PATIENTS


LOG_FILE_DIR = 'results'

if __name__ == "__main__":

    # Set log file
    import os
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, '_'.join(sys.argv[1:]))
    putil.LOG_FILE = LOG_FILE_PATH + '.txt'

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
    main = Task(
        name='RF', desc="Random Feature Support Vector Classifier").start()

    # Generate color map if flag set
    if args.get('knn'):
        ckm = CKM(
            args.get('k'),
            IDCDataset(PATIENTS, p=0.01, process=True, task=main.subtask()))
        idim = args.get('k')
        feature = ckm.map
    elif args.get('hsv'):
        from skimage import color
        idim = 7500

        def feature(image):
            return np.reshape(color.rgb2hsv(image), [-1])
    elif args.get('hue'):
        from skimage import color
        idim = 7500

        def feature(image):
            return np.reshape(color.rgb2hsv(image)[:, :, 0], [-1])
    else:
        idim = 7500
        feature = None

    # Make random feature
    rand_ft = make_feature(
        idim=idim, task=main,
        **args.subdict('ftype', 'kernel', 'fdim', 'cores'))

    # Load dataset
    dataset, validation_tester = make_trainset(
        transform=rand_ft,
        feature=ckm.map if args.get('knn') else None, main=main,
        **args.subdict('cores', 'ntrain', 'ptrain'))

    # Train model
    rfsvm = train(dataset, main.subtask())

    # Tester
    test_dataset, tester = make_testset(
        transform=rand_ft,
        feature=ckm.map if args.get('knn') else None, main=main,
        **args.subdict('cores', 'ntest', 'ptest'))

    # Run testers
    tester.loss(rfsvm, task=main)
    validation_tester.loss(rfsvm, task=main)

    # Save as JSON
    if os.path.exists(LOG_FILE_PATH + '.json'):
        main.acc_join()
        main.warn(
            "Specified log file path already exists. "
            "\"_1\" will be appended to end of file.")
        main.acc_join()
        if input('Overwrite? (Y/N) ') == 'Y':
            main.save(LOG_FILE_PATH + '.json')
    else:
        main.save(LOG_FILE_PATH + '.json')

    main.acc_join()
    main.done(desc="Program finished.")
