"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset


class Cwru(TensorDataset):
    """Cwru Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
    """
    domain_list = {
        "A": "A.txt",
        "B": "B.txt",
        "Sampled_B": "Sampled_B.txt",
        "Sampled_B_06": "Sampled_B_06.txt",
        "C": "C.txt",
        "D": "D.txt"
    }
    CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __init__(self, root: str, task: str, **kwargs):
        assert task in self.domain_list
        data_file = os.path.join(root, self.domain_list[task])

        row_data = torch.from_numpy(np.loadtxt(data_file, dtype=np.float32))
        x, y = row_data[:, 1:].view(row_data.shape[0], 1, 1, -1), row_data[:, 0].view(-1).type(torch.long)
        self.classes = Cwru.CLASSES
        super(Cwru, self).__init__(x, y)
