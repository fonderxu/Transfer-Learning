"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch
import torch.nn.functional as F

from common.utils.metric import binary_accuracy

__all__ = ['ImageClassifier']


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
                nn.Linear(100, num_classes + 1)
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
        output = predictions[:, :-1]
        d_output = torch.sigmoid(predictions[:, -1])
        if self.training:
            return f, (output, d_output.unsqueeze(1))
        else:
            return output

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


'''
def cross_entropy(output, target, weight=None):
    if weight is not None:
        return torch.mean(weight * F.cross_entropy(input=output, target=target, reduction='none'))
    return F.cross_entropy(output.log(), target)


def discriminator_eval(d_output_s, d_output_t, base_adv_weight=None, one_hot_label=None):
    d_label_s = torch.zeros(d_output_s.size(0)).to(d_output_s.device).type(torch.long)
    d_label_t = torch.ones(d_output_t.size(0)).to(d_output_t.device).type(torch.long)

    d_output = torch.cat((d_output_s, d_output_t), dim=0)
    d_label = torch.cat((d_label_s, d_label_t), dim=0)
    if base_adv_weight is None:
        return F.cross_entropy(d_output, d_label), binary_accuracy(d_output, d_label)
    else:
        weight = (one_hot_label * base_adv_weight.unsqueeze(dim=0)).sum(dim=1)
        return cross_entropy(d_output, d_label, weight), binary_accuracy(d_output, d_label)
'''


class ImageClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)


class WeightedDomainAdversarialLoss(nn.Module):
    def __init__(self, reduction: Optional[str] = 'mean'):
        super(WeightedDomainAdversarialLoss, self).__init__()
        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, d_output_s: torch.Tensor, d_output_t: torch.Tensor, weight=None) -> torch.Tensor:
        d_output = torch.cat((d_output_s, d_output_t), dim=0)
        d_label = torch.cat((
            torch.ones((d_output_s.size(0)), 1).to(d_output_s.device),
            torch.zeros((d_output_t.size(0)), 1).to(d_output_t.device),
        ))
        self.domain_discriminator_accuracy = binary_accuracy(d_output, d_label)
        """return loss of discriminator"""
        return self.bce(d_output, d_label, weight.view_as(d_output))


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

