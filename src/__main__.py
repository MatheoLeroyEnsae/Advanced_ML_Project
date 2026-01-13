"""
main of the package
"""

import random
import numpy as np
import torch


if __name__ == '__main__':
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)