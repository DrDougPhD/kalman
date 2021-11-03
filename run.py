import numpy

import kalman.dataset
from kalman import distributions


def main():
    # dataset = kalman.dataset.generate(columns=2, observations=100)
    # print(dataset)

    with kalman.dataset.generator(
            process_noise=1,
            measurement_noise=1,
            initial_state=numpy.array([[0, 1]]).T,
            state_to_measurement_transformation_matrix=numpy.array([
                [1, 0.1],
                [0, 1],
            ]),
            previous_state_transformation_matrix=numpy.array([[1, 2]]),
    ) as dataset:
        for i, observation in enumerate(dataset):
            print(observation)

            if i == 100:
                break


if __name__ == '__main__':
    main()
