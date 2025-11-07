import importlib.util
from typing import Union, TYPE_CHECKING

from xarray import DataArray, Dataset

if TYPE_CHECKING:
    from openeo import DataCube


def phenology(array: "Union[DataArray, DataCube]") -> Dataset:
    """
    Computes phenology metrics based on the `Phenolopy <https://github.com/lewistrotter/PhenoloPy>`_ implementation.


    Args:
        array: A DataArray containing a vegetation index

    Returns:
        A Dataset with multiple variables corresponding to phenometrics

    """
    from ._phenolopy import calc_phenometrics

    return calc_phenometrics(array)
