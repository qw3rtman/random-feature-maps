## Fast Random Kernelized Features
*Support Vector Machine Classification for High-Dimensional IDC Dataset*

## About
* [Presentation](https://github.com/qw3rtman/rf/blob/master/presentation.pdf)
* Paper (in-progress)

Random feature maps provide low-dimensional kernel approximations, thereby accelerating the training of support vector machines for large-scale datasets. Through a process of dimensionality reduction via k-means clustering, lifting via a random feature map, and subsequent linear SVM classification in this feature space, we outperform standard Gaussian kernel SVM on the high-dimensional invasive ductal carcinoma dataset (7500 dimensions) in both accuracy and speed. We explore applying two random maps (random Fourier features and random binning features) and experiment with different pre-processing methods such as k-means clustering, HSV transform, and histogram of gradients.

## Authors
[Tianshu Huang](http://tianshu.io/) | [Nimit Kalra](https://nimit.io/)
--- | ---
<tianshu.huang@utexas.edu> | <nimit@utexas.edu>
Department of Electrical and Computer Engineering | Department of Computer Science
University of Texas at Austin | University of Texas at Austin

## Usage
Run the module with ```python rf.py arg1=value arg2=value ...```, where parameters are passed as keyword arguments joined by '=' and separated by spaces.

Parameter |  Default | Description
--- | --- | ---
ptrain | 0.01 | Percent of training images to use
ptest | 0.1 | Percent of test images to use
fdim | 5000 | Feature space dimensionality
knn | False | Use Color k-means?
k | 5 | Number of means to use
kernel | G | Kernel type ("G", "L", or "C")
ntrain | -25 | Number of patients used for training
ntest | 25 | Number of patients used for testing
cores | 8 | Number of cores (processes) to use
ftype | F | Type of feature to use ("F" or "B")

The terminal output is mirrored to the file ```results_<command line args>.txt```.

## Example Session
Output file ```results_ptrain=0.1_ptest=1_fdim=2500.txt``` of ``` python rf.py ptrain=0.1 ptest=1 fdim=2500```:
```shell
----------------------------------------------------------------------------------------------------------

                                 Random Feature Support Vector Classifier

----------------------------------------------------------------------------------------------------------

    +---------+-----+------------------------------------------------------------+
    |Parameter|Value|Description                                                 |
    +---------+-----+------------------------------------------------------------+
    |ptrain   |0.1  |Percent of training images to use (default=0.01)            |
    +---------+-----+------------------------------------------------------------+
    |ptest    |1.0  |Percent of test images to use (default=0.1)                 |
    +---------+-----+------------------------------------------------------------+
    |fdim     |2500 |Feature space dimensionality (default=5000)                 |
    +---------+-----+------------------------------------------------------------+
    |knn      |False|Use Color k-means? (default=False)                          |
    +---------+-----+------------------------------------------------------------+
    |k        |5    |Number of means to use (default=5)                          |
    +---------+-----+------------------------------------------------------------+
    |kernel   |G    |Kernel type ("G", "L", or "C") (default=G)                  |
    +---------+-----+------------------------------------------------------------+
    |ntrain   |-25  |Number of patients used for training (default=-25)          |
    +---------+-----+------------------------------------------------------------+
    |ntest    |25   |Number of patients used for testing (default=25)            |
    +---------+-----+------------------------------------------------------------+
    |cores    |8    |Number of cores (processes) to use (8 available) (default=8)|
    +---------+-----+------------------------------------------------------------+
    |ftype    |F    |Type of feature to use ("F" or "B") (default=F)             |
    +---------+-----+------------------------------------------------------------+


-- Main Program ------------------------------------------------------------------------------------------
<RF> Random Feature Support Vector Classifier
[0.94s | 75.02MB] <Random Fourier Feature> 7500->2500 Random Fourier Feature created
  | <test data> Loading Images...
  |   | <IDC Dataset> started accounting
  |   |   | <Loader> Loading patient 10253...
  |   | [0.27s | 1.10MB] <Loader> loaded patient 10253
  |   |   | <Loader> Loading patient 10254...
  |   |   | <Loader> Loading patient 10261...

  			[Details omitted]

  |   | [6.37s | 4.66MB] <Loader> loaded patient 13693
  |   |   | <Loader> Loading patient 13694...
  |   | <IDC Dataset> stopped accounting
  | [113.93s | 502.89MB] <IDC Dataset> 25143 images (10.0%) sampled from 254 patients
  | <training data> Loading Images...
  |   | <IDC Dataset> started accounting
  |   |   | <Loader> Loading patient 9257...
  |   |   | <Loader> Loading patient 9258...

  			[Details omitted]

  |   | <IDC Dataset> stopped accounting
  | [84.02s | 516.57MB] <IDC Dataset> 25827 images (100.0%) sampled from 25 patients
  |   | <RF SVC> Computing RF SVC Classifier
  |   | [8.40s] <RF SVC> RFF SVC Computed
<Task> Classification experiment on new patients
  | <Tester> True Negative: 18126
  | <Tester> True Positive: 4116
  | <Tester> False Negative: 1588
  | <Tester> False Positive: 1997
  | <Tester> Total Tests: 25827
  | <Tester> Incorrect Tests: 3585.0
  | <Tester> Correct Tests: 22242.0
  | <Tester> Percent Error: 13.880822395167847
  | <Tester> Constant Classifier Baseline: 22.08541448871336
  | <Tester> Relative Improvement over Constant: 0.3714936886395512
[0.36s] <Tester> Done running tests.
  | <Tester> Time per test: 1.378298848930882e-05s
<Task> Classification verification on training data
  | <Tester> True Negative: 16185
  | <Tester> True Positive: 4704
  | <Tester> False Negative: 2715
  | <Tester> False Positive: 1539
  | <Tester> Total Tests: 25143
  | <Tester> Incorrect Tests: 4254.0
  | <Tester> Correct Tests: 20889.0
  | <Tester> Percent Error: 16.919222049874715
  | <Tester> Constant Classifier Baseline: 29.507218708984606
  | <Tester> Relative Improvement over Constant: 0.42660735948241
[0.35s] <Tester> Done running tests.
  | <Tester> Time per test: 1.3938018597950777e-05s
[213.03s] <RF> Program finished.
```

## Dependencies
* numpy
* scikit-learn
* scikit-image
* [print](https://github.com/thetianshuhuang/print)
* [syllabus](https://github.com/thetianshuhuang/syllabus)

## References
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
