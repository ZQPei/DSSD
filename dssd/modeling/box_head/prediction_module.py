import torch
import torch.nn as nn


class PredictionModule_C(nn.Module):
    def __init__(self, out_channels, boxes_per_location, num_classes):
        super(PredictionModule_C, self).__init__()
        # conv layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(out_channels, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
        )
        self.shortcut = nn.Conv2d(out_channels, 1024, kernel_size=1, stride=1, padding=0)

        self.cls_layer = nn.Conv2d(1024, boxes_per_location*num_classes, kernel_size=1, stride=1, padding=0)
        self.loc_layer = nn.Conv2d(1024, boxes_per_location*4, kernel_size=1, stride=1, padding=0)

    
    def forward(self, x):
        y = self.conv_layer(x) + self.shortcut(x)
        y_cls, y_loc = self.cls_layer(y), self.loc_layer(y)

        return y_cls, y_loc

