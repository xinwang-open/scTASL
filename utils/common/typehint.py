r"""
Type hint definitions
"""

from typing import Optional, TypeVar, Union

import h5py
import numpy as np
import scipy.sparse
from anndata.abc import CSCDataset, CSRDataset

Array = Union[np.ndarray, scipy.sparse.spmatrix]
BackedArray = Union[h5py.Dataset, CSRDataset, CSCDataset]
AnyArray = Union[Array, BackedArray]
RandomState = Optional[np.random.RandomState | int]

T = TypeVar("T")  # Generic type var
