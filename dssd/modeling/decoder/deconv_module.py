import torch
from torch import nn


class DeconvolutionModule(nn.Module):
    def __init__(self, cin_conv=1024, cin_deconv=512, cout=512, norm_layer=nn.BatchNorm2d, elementwise_type="sum", deconv_kernel_size=2, deconv_out_padding=0):
        super(DeconvolutionModule, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(cin_conv, cout, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            norm_layer(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            norm_layer(cout),
        )

        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(cin_deconv, cout, kernel_size=deconv_kernel_size, stride=2, padding=0, output_padding=deconv_out_padding),
            nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, dilation=1),
            norm_layer(cout)
        )

        if elementwise_type in ["sum", "prod"]:
            self.elementwise_type = elementwise_type
        else:
            raise RuntimeError("elementwise type incorrect!")
        self.relu = nn.ReLU(inplace=True)

        self.reset_parameters()


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    
    def forward(self, x_deconv, x_conv):
        y_deconv = self.deconv_layer(x_deconv)
        y_conv = self.conv_layer(x_conv)
        if self.elementwise_type == "sum":
            return self.relu(y_deconv + y_conv)
        elif self.elementwise_type == "prod":
            return self.relu(y_deconv + y_conv)
