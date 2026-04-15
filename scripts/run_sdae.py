from churn.experiments import ExperimentConfig


def main() -> None:
    config = ExperimentConfig(name="sdae_ensemble")
    print(config)


if __name__ == "__main__":
    main()
