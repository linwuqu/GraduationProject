from churn.experiments import ABLATION_GRID


def main() -> None:
    for row in ABLATION_GRID:
        print(row)


if __name__ == "__main__":
    main()
