import numpy

import kalman.dataset
import matplotlib.pyplot as plt


def main():
    numpy.random.seed(0)

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
        process_noise=numpy.array([
            [0.21, 0.1],
            [0.5, 2]
        ]),
        measurement_noise=numpy.array([
            [0.1, 0.1111],
            [0.53, 2]
        ]),
        initial_state_estimate=numpy.array([[0, 0]]).T,
        state_transformation_matrix=None,
        control_input_transformation_matrix=None,
        measurement_transformation_matrix=None,
        initial_estimate_error=None,
    ) as filter:
        state_predictions = None
        figure, (axis_1, axis_2) = plt.subplots(2, 1, sharex=True)
        axis_1.set_title('First Variable')
        axis_2.set_title('Second Variable')
        axis_2.set_xlabel('Time Step')

        upper_limits = numpy.array([[], []])
        lower_limits = numpy.array([[], []])
        for i, observation in enumerate(dataset):
            filter.predict()
            posterior = filter.correct(measurement=observation)

            state_predictions = posterior \
                if state_predictions is None \
                else numpy.hstack((state_predictions, posterior))

            _95th_prediction_interval = numpy.quantile(
                state_predictions,
                (0.025, 0.975),
                axis=1
            )
            print('-' * 120)
            print(lower_limits)
            print(_95th_prediction_interval[:1, :])
            print('-' * 120)
            lower_limits = numpy.hstack((lower_limits, _95th_prediction_interval[0:1, :].T))
            upper_limits = numpy.hstack((upper_limits, _95th_prediction_interval[1:2, :].T))

            axis_1.plot(i, observation[0], '+', color='tab:red')
            axis_2.plot(i, observation[1], '+', color='tab:red')


            # TODO: plot the bounds
            print('-' * 120)
            if i == 100:
                break

        axis_1.plot(dataset.time_steps, dataset.true_states[0], '-', color='black')
        axis_1.fill_between(
            dataset.time_steps,
            lower_limits[0],
            upper_limits[0],
            alpha=0.2
        )
        axis_2.plot(dataset.time_steps, dataset.true_states[1], '-', color='black')
        axis_2.fill_between(
            dataset.time_steps,
            lower_limits[1],
            upper_limits[1],
            alpha=0.2
        )

        plt.savefig('simulation.png')


if __name__ == '__main__':
    main()
