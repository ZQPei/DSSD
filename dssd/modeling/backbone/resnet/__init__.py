from ... import registry
from . import resnet

__all__ = ['resnet101']


@registry.BACKBONES.register('resnet101')
def resnet101(cfg, pretrained=True):
    return resnet.resnet101(pretrained=pretrained)
