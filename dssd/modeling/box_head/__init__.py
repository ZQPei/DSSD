from dssd.modeling import registry
from .box_head import DSSDBoxHead

__all__ = ['build_box_head', 'DSSDBoxHead']


def build_box_head(cfg):
    return registry.BOX_HEADS[cfg.MODEL.BOX_HEAD.NAME](cfg)
