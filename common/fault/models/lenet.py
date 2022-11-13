"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn


class LeNet(nn.Sequential):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__(
            nn.Conv2d(1, 20, kernel_size=(1, 5)),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=(1, 5)),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(50 * 253, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.num_classes = num_classes
        self.out_features = 1024

    def copy_head(self):
        return nn.Linear(500, self.num_classes)


def lenet(pretrained=False, **kwargs):
    """LeNet model from
    `"Gradient-based learning applied to document recognition" <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_

    Args:
        num_classes (int): number of classes. Default: 10

    .. note::
        The input image size must be 28 x 28.

    """
    return LeNet(**kwargs)
