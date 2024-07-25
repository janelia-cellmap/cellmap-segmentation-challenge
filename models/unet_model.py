# Link to source code
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py


""" Full assembly of the parts to form the complete network """

from .unet_parts import *

# from torch import utils


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, trilinear)
        self.up2 = Up(512, 256 // factor, trilinear)
        self.up3 = Up(256, 128 // factor, trilinear)
        self.up4 = Up(128, 64, trilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    # def use_checkpointing(self):
    #     self.inc = utils.checkpoint(self.inc)
    #     self.down1 = utils.checkpoint(self.down1)
    #     self.down2 = utils.checkpoint(self.down2)
    #     self.down3 = utils.checkpoint(self.down3)
    #     self.down4 = utils.checkpoint(self.down4)
    #     self.up1 = utils.checkpoint(self.up1)
    #     self.up2 = utils.checkpoint(self.up2)
    #     self.up3 = utils.checkpoint(self.up3)
    #     self.up4 = utils.checkpoint(self.up4)
    #     self.outc = utils.checkpoint(self.outc)
