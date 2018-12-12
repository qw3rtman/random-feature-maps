
import numpy as np
from multiprocessing import RawArray


def make_raw(array):
    """Create a multiprocessing-ready RawArray; creates a copy of ``array``.

    Parameters
    ----------
    array : np.array
        Array to use. Is copied into a RawArray.

    Returns
    -------
    (np.array, RawArray)
        [0] Numpy array created from raw_array. Do not share with other
            processes.
        [1] Raw Array; safe for sharing.

    References
    ----------
    https://research.wmz.ninja/articles/2018/03/
    on-sharing-large-arrays-when-using-pythons-multiprocessing.html
    """

    raw_array = RawArray('f', int(np.prod(array.shape)))
    array_np = np.frombuffer(raw_array, dtype=np.float32).reshape(array.shape)
    np.copyto(array_np, array)

    return array_np, raw_array
