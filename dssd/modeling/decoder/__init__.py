from dssd.modeling import registry
from .decoder import DSSDDecoder

__all__ = ['build_decoder', 'DSSDDecoder']


def build_decoder(cfg):
    return registry.DECODERS[cfg.MODEL.DECODER.NAME](cfg)