"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset


class Jnu(TensorDataset):
    """Gearbox Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
    """
    domain_list = {
        "A": "A.txt",
        "B": "B.txt",
        "C": "C.txt"
    }
    CLASSES = ['0', '1', '2', '3']

    def __init__(self, root: str, task: str, **kwargs):
        assert task in self.domain_list
        data_file = os.path.join(root, self.domain_list[task])

        row_data = torch.from_numpy(np.loadtxt(data_file, dtype=np.float32))
        x, y = row_data[:, :-1].view(row_data.shape[0], 1, 1, -1), row_data[:, -1].view(-1).type(torch.long)
        self.classes = Jnu.CLASSES
        super(Jnu, self).__init__(x, y)
