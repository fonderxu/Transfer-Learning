"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.modules.classifier import Classifier as ClassifierBase
from common.utils.metric import binary_accuracy
from ..modules.grl import WarmStartGradientReverseLayer


__all__ = ['ConditionalDomainAdversarialLoss', 'ImageClassifier']


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


class ConditionalDomainAdversarialLoss(nn.Module):
    r"""The Conditional Domain Adversarial Loss used in `Conditional Adversarial Domain Adaptation (NIPS 2018) <https://arxiv.org/abs/1705.10667>`_

    Conditional Domain adversarial loss measures the domain discrepancy through training a domain discriminator in a
    conditional manner. Given domain discriminator :math:`D`, feature representation :math:`f` and
    classifier predictions :math:`g`, the definition of CDAN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) &= \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(T(f_i^s, g_i^s))] \\
        &+ \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(T(f_j^t, g_j^t))],\\

    where :math:`T` is a :class:`MultiLinearMap`  or :class:`RandomizedMultiLinearMap` which convert two tensors to a single tensor.

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of
          features. Its input shape is (N, F) and output shape is (N, 1)
        entropy_conditioning (bool, optional): If True, use entropy-aware weight to reweight each training example.
          Default: False
        randomized (bool, optional): If True, use `randomized multi linear map`. Else, use `multi linear map`.
          Default: False
        num_classes (int, optional): Number of classes. Default: -1
        features_dim (int, optional): Dimension of input features. Default: -1
        randomized_dim (int, optional): Dimension of features after randomized. Default: 1024
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    .. note::
        You need to provide `num_classes`, `features_dim` and `randomized_dim` **only when** `randomized`
        is set True.

    Inputs:
        - g_s (tensor): unnormalized classifier predictions on source domain, :math:`g^s`
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - g_s, g_t: :math:`(minibatch, C)` where C means the number of classes.
        - f_s, f_t: :math:`(minibatch, F)` where F means the dimension of input features.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, )`.

    Examples::

        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
        >>> import torch
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim * num_classes, hidden_size=1024)
        >>> loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(g_s, f_s, g_t, f_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean'):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)

        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor, weight=None) -> torch.Tensor:
        f = torch.cat((f_s, f_t), dim=0)
        h = self.grl(f)
        d = self.domain_discriminator(h)
        d_label = torch.cat((
            torch.ones((f_s.size(0), 1)).to(f_s.device),
            torch.zeros((f_t.size(0), 1)).to(f_t.device),
        ))
        self.domain_discriminator_accuracy = binary_accuracy(d, d_label)
        """return loss of discriminator"""
        return self.bce(d, d_label, weight)


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
