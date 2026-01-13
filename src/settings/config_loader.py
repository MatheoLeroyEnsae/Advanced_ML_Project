"""Read and convert YAML into a DictConfig or ListConfig
Methods:
    load_config_from_file: Load a configuration file.
"""

from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf
from cloudpathlib import AnyPath

Config = ListConfig | DictConfig


def load_config_from_file(path: str | Path) -> Config:
    any_path = AnyPath(path)
    config: Config = OmegaConf.create(any_path.read_text())
    return config