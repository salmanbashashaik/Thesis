"""
Reproducibility utilities for setting random seeds.
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Allow CuDNN to select optimal algorithms (faster but non-deterministic)
    torch.backends.cudnn.benchmark = True


def get_generator(seed=42):
    """
    Get PyTorch random number generator with fixed seed.
    
    Args:
        seed: Random seed
        
    Returns:
        torch.Generator
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
