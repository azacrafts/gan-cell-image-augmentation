"""Mask-conditioned image synthesis (Pix2Pix-style) for LIVECell cGAN-Seg–aligned augmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, down: bool = True, norm: bool = True, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=not norm) if down else nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1, bias=not norm),
        ]
        if norm:
            layers.append(nn.BatchNorm2d(ch_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetGenerator(nn.Module):
    """
    U-Net generator: semantic one-hot mask (B, C, H, W) -> RGB phase (B, 3, H, W) in [0, 1] after sigmoid.
    """

    def __init__(self, mask_channels: int = 9, base: int = 64) -> None:
        super().__init__()
        # Encoder
        self.e1 = nn.Conv2d(mask_channels, base, 4, 2, 1)
        self.e2 = UNetBlock(base, base * 2, down=True)
        self.e3 = UNetBlock(base * 2, base * 4, down=True)
        self.e4 = UNetBlock(base * 4, base * 8, down=True)
        self.e5 = UNetBlock(base * 8, base * 8, down=True)
        self.e6 = UNetBlock(base * 8, base * 8, down=True)
        self.e7 = UNetBlock(base * 8, base * 8, down=True)
        self.e8 = UNetBlock(base * 8, base * 8, down=True, norm=False)
        # 8 stride-2 downs: use image_size >= 512 (bottleneck 2x2 after e8)

        # Decoder with skips (standard Pix2Pix)
        self.d8 = UNetBlock(base * 8, base * 8, down=False, dropout=0.5)
        self.d7 = UNetBlock(base * 8 * 2, base * 8, down=False, dropout=0.5)
        self.d6 = UNetBlock(base * 8 * 2, base * 8, down=False, dropout=0.5)
        self.d5 = UNetBlock(base * 8 * 2, base * 8, down=False)
        self.d4 = UNetBlock(base * 8 * 2, base * 4, down=False)
        self.d3 = UNetBlock(base * 4 * 2, base * 2, down=False)
        self.d2 = UNetBlock(base * 2 * 2, base, down=False)
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        s5 = self.e5(s4)
        s6 = self.e6(s5)
        s7 = self.e7(s6)
        s8 = self.e8(s7)

        u8 = self.d8(s8)
        u7 = self.d7(torch.cat([u8, s7], 1))
        u6 = self.d6(torch.cat([u7, s6], 1))
        u5 = self.d5(torch.cat([u6, s5], 1))
        u4 = self.d4(torch.cat([u5, s4], 1))
        u3 = self.d3(torch.cat([u4, s3], 1))
        u2 = self.d2(torch.cat([u3, s2], 1))
        u1 = self.d1(torch.cat([u2, s1], 1))
        return u1


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator: concat(mask, image) -> logits map."""

    def __init__(self, mask_channels: int = 9, image_channels: int = 3, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        ch = mask_channels + image_channels
        layers: list[nn.Module] = [nn.Conv2d(ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, True),
            ]
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [nn.Conv2d(nf_prev, nf, 4, 1, 1, bias=False), nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(nf, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([mask, image], dim=1))


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
