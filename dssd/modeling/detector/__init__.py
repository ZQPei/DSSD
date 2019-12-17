from .dssd_detector import DSSDDetector

_DETECTION_META_ARCHITECTURES = {
    "DSSDDetector": DSSDDetector
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
