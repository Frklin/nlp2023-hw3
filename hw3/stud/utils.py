import os
import random
import numpy as np
import torch
import config



def seed_everything(seed=42):
    """
    Seeds basic parameters for reproductibility of results

    Args:
        seed (int, optional): Number of the seed. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed) # if you are using GPU
    # torch.backends.cudnn.deterministic = True  # if you are using GPU
    # torch.backends.cudnn.benchmark = False



def collate_fn(batch):
    pass