"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn


class AlexNet(nn.Sequential):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__(
            nn.Conv2d(1, 64, kernel_size=(1, 32), stride=(1, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Conv2d(64, 192, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Conv2d(192, 384, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Flatten(start_dim=1)
        )

        self.num_classes = num_classes
        self.out_features = 256 * 1 * 3

    def copy_head(self):
        return nn.Linear(256 * 1 * 3, self.num_classes)


def alexnet(pretrained=False, **kwargs):
    """AlexNet model from
    `"Gradient-based learning applied to document recognition" <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_

    Args:
        num_classes (int): number of classes. Default: 10

    .. note::
        The input image size must be 28 x 28.

    """
    return AlexNet(**kwargs)
