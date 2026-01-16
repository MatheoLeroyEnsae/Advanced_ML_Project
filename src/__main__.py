"""
main of the package
"""

import random
import numpy as np
import torch
from src.applications.generate_dataset import job
from src.applications.uncertainty_quantification import job_uncertainty
from src.settings.config_loader import load_config_from_file
from pathlib import Path


if __name__ == '__main__':
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ROOT_DIR = Path(__file__).resolve().parent.parent
    CONF_DIR = ROOT_DIR / "conf"

    CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
    path = CONF_DIR / "conf.yaml"

    if not CONF_DIR.exists():
        raise FileNotFoundError(f"Config directory not found: {CONF_DIR}")

    config = load_config_from_file(path)
    train_generations, validation_generations, results_dict = job(config=config, return_bool=True)
    results_dict, entropies = job_uncertainty(
        train_generations, validation_generations, results_dict, config)
