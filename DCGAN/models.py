import torch
import torch.nn as nn
import torch.nn.init as init

class Generator(nn.Module):
    def __init__(self, latent_dim, channels=3, img_size=64):
        super(Generator, self).__init__()

        def deconvolution_block(in_channels, out_channels, kernel_size, stride,
                                padding, bn=True):
            block = [nn.ConvTranspose2d(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(out_channels, 0.8))
            block.extend([
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.25)
            ])
            return nn.Sequential(*block)

        self.conv_blocks = nn.Sequential(
            deconvolution_block(latent_dim, img_size * 8, 4, 1, 0),
            deconvolution_block(img_size * 8, img_size * 4, 4, 2, 1),
            deconvolution_block(img_size * 4, img_size * 2, 4, 2, 1),
            deconvolution_block(img_size * 2, img_size, 4, 2, 1),
            nn.ConvTranspose2d(img_size, channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        img = self.conv_blocks(x)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, channels=3, img_size=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ])
            return nn.Sequential(*block)

        self.model = nn.Sequential(
            discriminator_block(channels, img_size, bn=False),
            discriminator_block(img_size, img_size * 2),
            discriminator_block(img_size * 2, img_size * 4),
            discriminator_block(img_size * 4, img_size * 8),
        )

        ds_size = img_size // 2 ** 4
        self.flatten = nn.Flatten()
        self.adv_layer = nn.Linear(img_size * 8 * ds_size ** 2, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.zero_()

    def forward(self, img):
        out = self.model(img)
        out = self.flatten(out)
        validity = self.adv_layer(out)
        return validity