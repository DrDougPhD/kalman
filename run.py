import numpy

import kalman.dataset
from kalman import distributions


def main():
    initial_state = numpy.array([[0, 0.1]]).T

    with kalman.filter() as filter, kalman.dataset.generator(
            process_noise=1,
            measurement_noise=1,
            initial_state=initial_state,
            state_estimate_transformation_matrix=numpy.array([
                [1, 0.1],
                [0, 1],
            ]),
            measurement_transformation_matrix=numpy.diag((1, 2)),
    ) as dataset:
        for i, observation in enumerate(dataset):
            print(observation)

            if i == 100:
                break


if __name__ == '__main__':
    main()
