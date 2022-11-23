import pathlib

from eta_utility import LOG_INFO, get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger(level=LOG_INFO, format="logname")
    experiment(pathlib.Path(__file__).parent)


def experiment(root_path, overwrite=None):
    """Perform an experiment with the MPC algorithm and the cleaning machine environment.

    :param root_path: Root path of the experiment.
    :param overwrite: Additional config values to overwrite values from JSON.
    """
    experiment = ETAx(
        root_path=root_path, config_overwrite=overwrite, relpath_config=".", config_name="config_experiment"
    )
    experiment.play("CPSL2023", "exp_221118")


if __name__ == "__main__":
    main()
