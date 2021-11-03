from typing import Callable
from typing import Mapping
from typing import Tuple

import numpy
import numpy.typing

from . import types


class RandomVariable(object):
    random_number_generator: Callable
    distribution_parameters: Mapping[str, float]

    def __init__(self, dimension: int, range: Tuple[float, float] = None, inclusivity: Tuple[bool, bool] = None):
        self.lower_bound, self.upper_bound = range or (0, 1)
        self.lower_bound_inclusive, self.upper_bound_inclusive = inclusivity or (False, False)
        self.dimension = dimension

    def __iter__(self):
        return self

    def __next__(self) -> types.Matrix:
        return self.random_number_generator(**self.distribution_parameters)


class Normal(RandomVariable):
    def __init__(self, mean: float, standard_deviation: float, **kwargs):
        super().__init__(**kwargs)
        self.random_number_generator = numpy.random.normal
        self.distribution_parameters = {
            'loc': mean,
            'scale': standard_deviation,
            'size': (self.dimension, 1)
        }


class StandardNormal(Normal):
    def __init__(self, **kwargs):
        super().__init__(mean=0, standard_deviation=1, **kwargs)
