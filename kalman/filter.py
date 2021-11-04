from contextlib import contextmanager
from typing import Optional

import numpy
import numpy.linalg

from . import types


class DiscreteKalmanFilter(object):
    _process_noise: types.Matrix
    _measurement_noise: types.Matrix

    previous_corrected_state: types.Matrix
    a_priori_state_projection: types.Matrix = None
    a_priori_estimate_error: types.Matrix = None

    kalman_gain: types.Matrix = None
    a_posteriori_estimate_error: types.Matrix

    _state_transformation_matrix: types.Matrix
    _measurement_transformation_matrix: types.Matrix
    _control_input_transformation_matrix: types.Matrix

    def __init__(self,
                 process_noise: types.Matrix,
                 measurement_noise: types.Matrix,
                 initial_state_estimate: types.Matrix,
                 state_transformation_matrix: Optional[types.Matrix] = None,
                 measurement_transformation_matrix: Optional[types.Matrix] = None,
                 control_input_transformation_matrix: Optional[types.Matrix] = None,
                 initial_estimate_error: Optional[types.Matrix] = None,
                 ):
        state_dimension = initial_state_estimate.shape[0]
        self.I = numpy.identity(state_dimension)

        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self.previous_corrected_state = initial_state_estimate
        self.a_posteriori_estimate_error = initial_estimate_error \
            if initial_estimate_error is not None \
            else self.I

        self._state_transformation_matrix = state_transformation_matrix \
            if state_transformation_matrix is not None \
            else self.I
        self._measurement_transformation_matrix = measurement_transformation_matrix \
            if measurement_transformation_matrix is not None \
            else self.I
        self._control_input_transformation_matrix = control_input_transformation_matrix \
            if control_input_transformation_matrix is not None \
            else numpy.zeros((state_dimension, 1))

        self.step_counter = 0

    def predict(self, control_input: Optional[types.Matrix] = None) -> types.Matrix:
        """
        Time-update / projection step of the Kalman filter.
        :param control_input: Optional input to control the outcome state.
        :return: A priori state projection for current time step.
        """
        print(f'Predicting state ad step #{self.step_counter}')

        control_input = control_input if control_input is not None else numpy.zeros((1, 1))

        self.a_priori_state_projection = self._project_next_state(control_input=control_input)
        self.a_priori_estimate_error = self._predict_next_error_covariance()

        print('\tState projection:')
        print(self.a_priori_state_projection)
        print('\tError estimate:')
        print(self.a_priori_estimate_error)

        return self.a_priori_state_projection

    def _project_next_state(self, control_input: types.Matrix) -> types.Matrix:
        """
        Step 1 of the time update / projection phase of the Kalman filter, where the current state is estimated before
        a new measurement is taken, given the previous state.
        :param control_input: An optional input controlling a modification to the previous state.
        :return: The a priori state projection for the current time step.
        """
        return (
            self._state_transformation_matrix @ self.previous_corrected_state
        ) + (
            self._control_input_transformation_matrix @ control_input
        )

    def _predict_next_error_covariance(self) -> types.Matrix:
        """
        Step 2 of the time update / projection phase of the Kalman filter, where the current estimate error is estimated
        given the previous estimated error.
        :return: The a priori estimate error for the current state projection.
        """
        A = self._state_transformation_matrix
        previous_P = self.a_posteriori_estimate_error

        return (
            A @ previous_P @ A.T
        ) + self._process_noise

    def correct(self, measurement: types.Matrix) -> types.Matrix:
        """
        Measurement update / correction phase of the Kalman filter, where a new measurement is utilized to update the
        predicted state.
        :param measurement: The new measurement taken of the current state, with its inherent noise due to process noise
        and measurement noise.
        :return: The updated state estimate for the current step.
        """
        print(f'Correcting state estimate given measurement #{self.step_counter}')

        self.kalman_gain = self._compute_next_kalman_gain()
        updated_state = self._correct_state_estimate_given_measurement(measurement=measurement)
        self.a_posteriori_estimate_error = self._update_estimate_error()

        print(f'\tKalman gain:')
        print(self.kalman_gain)
        print('\tCorrected state:')
        print(updated_state)
        print('\tUpdated error:')
        print(self.a_posteriori_estimate_error)

        self.step_counter += 1
        self.previous_corrected_state = updated_state
        return updated_state

    def _compute_next_kalman_gain(self) -> types.Matrix:
        H = self._measurement_transformation_matrix
        a_priori_P = self.a_priori_estimate_error
        try:
            return (
                a_priori_P @ H.T
            ) @ numpy.linalg.inv(
                (H @ a_priori_P @ H.T) + self._measurement_noise
            )
        except numpy.linalg.LinAlgError:
            print('Invariant matrix cannot have its inverse computed!')
            print(a_priori_P)
            print(H @ a_priori_P @ H.T)
            print((H @ a_priori_P @ H.T) + self._measurement_noise)
            raise

    def _correct_state_estimate_given_measurement(self, measurement: types.Matrix) -> types.Matrix:
        a_priori_state = self.a_priori_state_projection
        return a_priori_state + (
            self.kalman_gain @ (
                measurement - (self._measurement_transformation_matrix @ a_priori_state)
            )
        )

    def _update_estimate_error(self) -> types.Matrix:
        return (
            self.I - (self.kalman_gain @ self._state_transformation_matrix)
        ) @ self.a_priori_estimate_error


@contextmanager
def initialize(**kwargs) -> DiscreteKalmanFilter:
    yield DiscreteKalmanFilter(**kwargs)
