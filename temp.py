import torch

from dssd.modeling.detector import DSSDDetector
from dssd.config import cfg


if __name__ == '__main__':
    dssd = DSSDDetector(cfg)
    dssd.eval()
    x = torch.randn(10, 3, 320, 320)
    y = dssd(x)