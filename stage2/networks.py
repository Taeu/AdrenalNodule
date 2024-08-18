import torch
import torch.nn as nn



def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

def class_head(in_channels, out_channels, n_class):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(2048, n_class),
        #nn.Sigmoid()
    )

class UNet_(nn.Module):

    def __init__(self, pretrained, n_class):
        super().__init__()

        self.pretrained = pretrained
        self.class_head = class_head(512,512, n_class)
    def forward(self, x):
        features = self.pretrained.encoder(x)
        decoder_output = self.pretrained.decoder(*features)

        masks = self.pretrained.segmentation_head(decoder_output)
        if self.class_head is not None:
            labels = self.class_head(features[-1])
            return masks, labels
        return masks


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.class_head = class_head(512, 256)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x) # 160
        x = self.maxpool(conv1) # 80

        conv2 = self.dconv_down2(x) # 80
        x = self.maxpool(conv2) # 40

        conv3 = self.dconv_down3(x) # 40
        x = self.maxpool(conv3) # 20

        x = self.dconv_down4(x) # 20
        out_class = self.class_head(x)
        x = self.upsample(x)

        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)

        return out, out_class