from ..modules.grl import WarmStartGradientReverseLayer
import torch.nn.functional as F

import torch.nn as nn
import torch

from common.utils.metric import binary_accuracy

"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch
import torch.nn.functional as F

from common.utils.metric import binary_accuracy

__all__ = ['FaultClassifier']


class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=False, pool_layer=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Identity()
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Sequential(
                nn.Linear(self._features_dim, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(100, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Linear(100, num_classes)
            )
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        if self.training:
            return f, predictions
        else:
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


class DomainAdversarial(nn.Module):
    def __init__(self):
        super(DomainAdversarial, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.bce = lambda input, target: F.binary_cross_entropy(input, target)
        self.domain_classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256 * 1 * 1, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        r_f_s = self.grl(f_s)
        r_f_t = self.grl(f_t)
        d_ouput_s = self.domain_classifier(r_f_s)
        d_ouput_t = self.domain_classifier(r_f_t)
        d = torch.cat((d_ouput_s, d_ouput_t), dim=0)
        d_label = torch.cat((
            torch.zeros((f_s.size(0))).to(f_s.device),
            torch.ones((f_t.size(0))).to(f_t.device),
        )).long()

        self.domain_discriminator_accuracy = domain_acccuacy(d, d_label)
        return F.cross_entropy(d, d_label)


class FaultClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 100, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, bottleneck_dim),
            nn.ReLU(),
        )
        super(FaultClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)


def domain_acccuacy(d_output, d_label):
    with torch.no_grad():
        batch = d_output.shape[0]
        pred = torch.softmax(d_output, dim=1).data.max(1)[1]  # get the index of the max log-probability
        correct = pred.eq(d_label.data.view_as(pred)).cpu().sum()
        return correct / batch * 100.
