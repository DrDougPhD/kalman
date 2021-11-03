from typing import Callable
from typing import Mapping
from typing import Tuple

import numpy
import numpy.typing


class RandomVariable(object):
    random_number_generator: Callable
    distribution_parameters: Mapping[str, float]

    def __init__(self, dimension: int, range: Tuple[float, float] = None, inclusivity: Tuple[bool, bool] = None):
        self.lower_bound, self.upper_bound = range or (0, 1)
        self.lower_bound_inclusive, self.upper_bound_inclusive = inclusivity or (False, False)
        self.dimension = dimension

    def __iter__(self) -> numpy.typing.NDArray[numpy.float64]:
        while True:
            yield self.random_number_generator(**self.distribution_parameters)


class Normal(RandomVariable):
    def __init__(self, mean: float, standard_deviation: float, **kwargs):
        super().__init__(**kwargs)
        self.random_number_generator = numpy.random.normal
        self.distribution_parameters = {
            'loc': mean,
            'scale': standard_deviation,
            'size': self.dimension
        }


class StandardNormal(Normal):
    def __init__(self, **kwargs):
        super().__init__(mean=0, standard_deviation=1, **kwargs)


if __name__ == '__main__':
    std_normal = StandardNormal(dimension=2)
    for i, value in enumerate(std_normal):
        print(value)

        if i == 50:
            break
