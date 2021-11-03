import itertools
from contextlib import contextmanager
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy
import numpy.typing

from . import distributions
from . import types


def generate(columns: int, observations: int) -> types.Matrix:
    return numpy.zeros(shape=(observations, columns))


@contextmanager
def generator(process_noise: float,
              measurement_noise: float,
              initial_state: types.Matrix,
              measurement_transformation_matrix: Optional[types.Matrix] = None,
              state_estimate_transformation_matrix: Optional[types.Matrix] = None,
              control_input: Optional[Tuple[types.Matrix, Callable[[], types.Matrix]]] = None
              ) -> types.Matrix:
    state_dimension = initial_state.shape[0]
    measurement_dimension = measurement_transformation_matrix.shape[0]

    yield DatasetGenerator(
        initial_state=initial_state,
        process_noise=distributions.Normal(mean=0,
                                           standard_deviation=process_noise,
                                           dimension=state_dimension),
        measurement_noise=distributions.Normal(mean=0,
                                               standard_deviation=measurement_noise,
                                               dimension=measurement_dimension),
        state_estimate_transformation_matrix=state_estimate_transformation_matrix,
        measurement_transformation_matrix=measurement_transformation_matrix,
        control_input=control_input
    )


class DatasetGenerator(object):
    def __init__(self, initial_state: types.Matrix,
                 process_noise: distributions.RandomVariable,
                 measurement_noise: distributions.RandomVariable,
                 state_estimate_transformation_matrix: Optional[types.Matrix] = None,
                 measurement_transformation_matrix: Optional[types.Matrix] = None,
                 control_input: Optional[Tuple[types.Matrix, Iterable[types.Matrix]]] = None,
                 ):
        self.previous_state = initial_state
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        state_dimension = initial_state.shape[0]
        I = numpy.identity(state_dimension)

        self.state_estimate_transformation_matrix = state_estimate_transformation_matrix \
            if state_estimate_transformation_matrix is not None \
            else I
        self.measurement_transformation_matrix = measurement_transformation_matrix \
            if measurement_transformation_matrix is not None \
            else I

        self.control_input_transformation_matrix, self.control_input_generator = control_input or (
            numpy.zeros((state_dimension, 1)),
            itertools.repeat(numpy.zeros((1, 1))),  # generates singleton matrix of 0s every time
        )

    def __iter__(self):
        return self

    def __next__(self) -> types.Matrix:
        new_state = numpy.dot(self.state_estimate_transformation_matrix, self.previous_state) \
                    + next(self.process_noise) \
                    + numpy.dot(self.control_input_transformation_matrix, next(self.control_input_generator))
        new_measurement = numpy.dot(self.measurement_transformation_matrix, new_state) \
                        + next(self.measurement_noise)

        self.previous_state = new_state

        return new_measurement