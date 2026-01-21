"""
Code to parse loading from a config YAML file.

"""

from pathlib import Path

import yaml


def load_config(config_file: str) -> dict:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    config : dict
        Configuration parameters as a dictionary.
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file {config_file} does not exist.')
    try:
        with open(config_path, 'r') as f:
            all_config = yaml.safe_load(f)
            if all_config is None:
                raise ValueError('Config file is empty.')
            return all_config
    except yaml.YAMLError as e:
        raise ValueError(f'Error parsing YAML config file: {e}')
