from dssd.modeling import registry
from .resnet import resnet101

__all__ = ['build_backbone', 'resnet101']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
