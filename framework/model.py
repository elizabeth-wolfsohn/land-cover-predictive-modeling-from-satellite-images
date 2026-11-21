"""
Segmentation models: U-Net benchmark and Swin Transformer + FPN decoder.
All models take (B, 4, 256, 256) images and return (B, num_classes, 256, 256) logits.
"""
import torch
from torch import nn
import torch.nn.functional as F
import timm
import torchvision
from huggingface_hub import hf_hub_download


def bn_conv_relu(in_ch, out_ch):
    return nn.Sequential(nn.BatchNorm2d(in_ch), nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))


def bn_upconv_relu(in_ch, out_ch):
    return nn.Sequential(nn.BatchNorm2d(in_ch),
                         nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
                         nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net (Ronneberger et al 2015), port of the TF benchmark
    (adapted from github.com/karolzak/keras-unet/master/keras_unet/models/satellite_unet.py)
    """
    def __init__(self, in_channels=4, num_classes=10, num_layers=4, filters=64, upconv_filters=96):
        super().__init__()
        f, uf = filters, upconv_filters
        self.stem = nn.Sequential(nn.Conv2d(in_channels, f, 3, padding=1), nn.ReLU(inplace=True))
        self.c1 = bn_conv_relu(f, f)
        self.pre_pool = bn_conv_relu(f, f)
        self.down = nn.ModuleList([nn.Sequential(bn_conv_relu(f, f), bn_conv_relu(f, f)) for _ in range(num_layers)])
        self.down_post = nn.ModuleList([bn_conv_relu(f, f) for _ in range(num_layers)])
        self.bottom = nn.Sequential(bn_conv_relu(f, f), bn_conv_relu(f, f), bn_upconv_relu(f, f))
        self.up = nn.ModuleList([
            nn.Sequential(bn_conv_relu(2*f, uf), bn_conv_relu(uf, f), bn_upconv_relu(f, f)) for _ in range(num_layers)
        ])
        self.last = nn.Sequential(bn_conv_relu(2*f, uf), bn_conv_relu(uf, f))
        self.head = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.c1(x)
        x = F.max_pool2d(self.pre_pool(c1), 2)
        skips = []
        for down, post in zip(self.down, self.down_post):
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(post(x), 2)
        x = self.bottom(x)
        for up, skip in zip(self.up, reversed(skips)):
            x = up(torch.cat([x, skip], 1))
        x = self.last(torch.cat([x, c1], 1))
        return self.head(x)


class SatlasSwin(nn.Module):
    """Swin-v2 backbone pretrained on Sentinel-2 by SatlasPretrain (github.com/allenai/satlaspretrain_models),
    multispectral single-image weights. Returns 4 NCHW feature maps (strides 4, 8, 16, 32).
    """
    ARCHS = {
        'satlas_swint_ms': (torchvision.models.swin_v2_t, 'sentinel2_swint_si_ms.pth', [96, 192, 384, 768]),
        'satlas_swinb_ms': (torchvision.models.swin_v2_b, 'sentinel2_swinb_si_ms.pth', [128, 256, 512, 1024]),
    }
    # Satlas MS input bands: R, G, B, B05, B06, B07, B08, B11, B12 -> our R, G, B, NIR(B08)
    BANDS = [0, 1, 2, 6]

    def __init__(self, name='satlas_swint_ms', pretrained=True):
        super().__init__()
        arch, weights_file, self.channels = self.ARCHS[name]
        swin = arch()
        embed = swin.features[0][0]
        swin.features[0][0] = nn.Conv2d(9, embed.out_channels, 4, stride=4)
        if pretrained:
            path = hf_hub_download('allenai/satlas-pretrain', weights_file)
            state = torch.load(path, map_location='cpu')
            prefix = 'backbone.backbone.'
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            missing, _ = swin.load_state_dict(state, strict=False)
            # only the unused ImageNet classification head is not in the checkpoint
            assert all(k.startswith(('norm.', 'head.')) for k in missing), missing
        # keep the pretrained patch embedding weights of the 4 bands we have
        # ponytail: the 5 missing bands (red-edge, SWIR) simply contribute nothing
        conv9 = swin.features[0][0]
        conv4 = nn.Conv2d(4, conv9.out_channels, 4, stride=4)
        conv4.weight.data = conv9.weight.data[:, self.BANDS].clone()
        conv4.bias.data = conv9.bias.data.clone()
        swin.features[0][0] = conv4
        self.features = swin.features

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (1, 3, 5, 7):  # outputs of the 4 stages (the others are patch embedding/merging)
                feats.append(x.permute(0, 3, 1, 2))
        return feats


class SwinFPN(nn.Module):
    """Pretrained backbone (timm model name, or a SatlasSwin name) with a light FPN decoder."""
    def __init__(self, backbone='swinv2_tiny_window8_256', in_channels=4, num_classes=10,
                 decoder_channels=128, pretrained=True):
        super().__init__()
        if backbone in SatlasSwin.ARCHS:
            self.encoder = SatlasSwin(backbone, pretrained=pretrained)
            channels, self.nhwc = self.encoder.channels, False
        else:
            # timm adapts the pretrained patch embedding to 4 channels (in_chans)
            self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True, in_chans=in_channels)
            # transformers output NHWC features, CNNs NCHW
            channels = self.encoder.feature_info.channels()
            self.nhwc = getattr(self.encoder, 'output_fmt', 'NCHW') == 'NHWC'
        d = decoder_channels
        self.lateral = nn.ModuleList([nn.Conv2d(c, d, 1) for c in channels])
        self.smooth = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(d, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.encoder(x)
        if self.nhwc:
            feats = [f.permute(0, 3, 1, 2) for f in feats]
        # top-down pathway: from the coarsest to the finest scale
        y = self.lateral[-1](feats[-1])
        for lat, f in zip(reversed(self.lateral[:-1]), reversed(feats[:-1])):
            y = lat(f) + F.interpolate(y, size=f.shape[-2:], mode='bilinear', align_corners=False)
        y = self.head(self.smooth(y))
        return F.interpolate(y, size=size, mode='bilinear', align_corners=False)


def build_model(config, pretrained=True):
    """Instantiate the model described by a train config."""
    if config.model == 'unet':
        return UNet(num_classes=10, num_layers=2)
    if config.model == 'swin':
        return SwinFPN(backbone=config.backbone, num_classes=10, pretrained=pretrained)
    raise ValueError(f"unknown model {config.model!r}")


if __name__ == '__main__':
    x = torch.randn(2, 4, 256, 256)
    for model in (UNet(num_layers=2), SwinFPN(), SwinFPN('satlas_swint_ms')):
        model.eval()
        with torch.no_grad():
            y = model(x)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{type(model).__name__}: {n_params:.1f}M params, output {tuple(y.shape)}")
        assert y.shape == (2, 10, 256, 256)
