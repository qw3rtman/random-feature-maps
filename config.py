
"""Program Configuration

Attributes
----------
CORES : int
    Number of cores
RF_PARAMS : Config
    Configuration class for random features
"""

from syllabus import Config, Param
import multiprocessing as mp


CORES = mp.cpu_count()

RF_PARAMS = Config(
    Param('ptrain', desc='Percent of training images to use',
          aliases=['train'], type=float, default=0.01),
    Param('ptest', desc='Percent of test images to use',
          aliases=['test'], type=float, default=0.1),
    Param('fdim', desc='Feature space dimensionality',
          aliases=['D', 'P'], type=int, default=5000),
    Param('knn', desc='Use Color K Means?',
          flags={'-knn': True}, type=bool, default=False),
    Param('hsv', desc='Use HSV transform?',
          flags={'-hsv': True}, type=bool, default=False),
    Param('hue', desc='Use Hue transform?',
          flags={'-hue': True}, type=bool, default=False),
    Param('k', desc='Number of means to use',
          aliases=['k', 'K'], type=int, default=5),
    Param('kernel', desc='Kernel type ("G", "L", or "C")',
          default='G'),
    Param('ntrain', desc='Number of patients used for training',
          default=-25, type=int),
    Param('ntest', desc='Number of patients used for testing',
          default=25, type=int),
    Param('cores', type=int, default=CORES,
          desc=('Number of cores (processes) to use ({n} available)'
                .format(n=CORES))),
    Param('ftype', desc='type of feature to use ("F" or "B")',
          default='F')
)
