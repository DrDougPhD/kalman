import numpy

import kalman.dataset


def main():
    initial_state = numpy.array([[0, 0.1]]).T

    with kalman.dataset.generator(
        process_noise=1,
        measurement_noise=1,
        initial_state=initial_state,
        state_estimate_transformation_matrix=numpy.array([
            [1, 0.1],
            [0, 1],
        ]),
        measurement_transformation_matrix=numpy.diag((1, 2)),
    ) as dataset, kalman.filter(
        process_noise=numpy.array([[0.1]]),
        measurement_noise=numpy.array([[0.1]]),
        initial_state_estimate=numpy.array([[0, 0]]).T,
        state_transformation_matrix=None,
        control_input=None,
        initial_estimate_error=None,
        measurement_transformation_matrix=None,
    ) as filter:
        for i, observation in enumerate(dataset):
            filter.predict()
            filter.correct(measurement=observation)

            if i == 100:
                break


if __name__ == '__main__':
    main()
