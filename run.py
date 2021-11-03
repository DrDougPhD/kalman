import kalman.dataset


def main():
    dataset = kalman.dataset.generate(columns=2, observations=100)
    print(dataset)


if __name__ == '__main__':
    main()
