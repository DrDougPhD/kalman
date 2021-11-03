import numpy
import numpy.typing


def generate(columns: int, observations: int) -> numpy.typing.NDArray[numpy.float64]:
    return numpy.zeros(shape=(observations, columns))
