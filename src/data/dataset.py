from torch.utils.data import Dataloader
from got10k.datasets import GOT10k


class Got10kDataLoader: 
    def __init__(self, root_dir='data/GOT-10k', subset='train') -> None:
        dataset = GOT10k(root_dir=root_dir, subset=subset)
        pass