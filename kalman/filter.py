import itertools
from contextlib import contextmanager
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy

from . import types


class DiscreteKalmanFilter(object):
    def __init__(self,
                 process_noise: types.Matrix,
                 measurement_noise: types.Matrix,
                 initial_state_estimate: types.Matrix,
                 state_transformation_matrix: Optional[types.Matrix] = None,
                 measurement_transformation_matrix: Optional[types.Matrix] = None,
                 control_input: Optional[Tuple[types.Matrix, Iterable[types.Matrix]]] = None,
                 initial_estimate_error: Optional[types.Matrix] = None,
                 ):
        state_dimension = initial_state_estimate.shape[0]
        I = numpy.identity(state_dimension)

        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.previous_estimated_state = initial_state_estimate
        self.previous_estimate_error = initial_estimate_error \
            if initial_estimate_error is not None \
            else I

        self.state_transformation_matrix = state_transformation_matrix \
            if state_transformation_matrix is not None \
            else I
        self.measurement_transformation_matrix = measurement_transformation_matrix \
            if measurement_transformation_matrix is not None \
            else I
        self.control_input = control_input or (
            numpy.zeros((state_dimension, 1)),
            itertools.repeat(numpy.zeros((1, 1))),  # generates singleton matrix of 0s every time
        )

        self.step_counter = 0

    def predict(self) -> types.Matrix:
        print(f'Predicting state ad step #{self.step_counter}')
        pass

    def correct(self, measurement: types.Matrix) -> types.Matrix:
        print(f'Correcting state estimate given measurement #{self.step_counter}')

        self.step_counter += 1
        pass


@contextmanager
def initialize(**kwargs) -> DiscreteKalmanFilter:
    yield DiscreteKalmanFilter(**kwargs)
