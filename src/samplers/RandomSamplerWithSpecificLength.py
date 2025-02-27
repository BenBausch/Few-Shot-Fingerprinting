import torch
from torch.utils.data import Sampler


class RandomSamplerWithSpecificLength(Sampler):
    def __init__(self, data_source, length):
        self.data_source = data_source
        self.length = length

    def __iter__(self):
        n = len(self.data_source)
        indices = torch.randint(high=n, size=(self.length,), dtype=torch.int64)
        return iter(indices.tolist())

    def __len__(self):
        return self.length
