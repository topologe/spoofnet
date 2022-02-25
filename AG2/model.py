from torch import nn


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AG2Config:
    def __init__(self, args):
        self.image_size = args['image_size']  # Size of feature maps in generator
        self.num_channels = args['num_channels']  # Number of channels in the training images.
        self.num_filters = args['num_filters']
        self.num_layers = args['num_layers']
        self.kernel_size = (args['kernel_size'], args['kernel_size'])
        self.stride = (args['stride'], args['stride'])
        self.padding = args['padding']


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False, batch_norm=True, activation='relu'):
        super().__init__()
        self.batch_norm = batch_norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False, batch_norm=True, activation='relu'):
        super().__init__()
        self.batch_norm = batch_norm

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x


class AG2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        encoder_layers = [EncoderBlock(config.num_channels, config.num_filters, config.kernel_size, config.stride, config.padding)]
        encoder_layers += [EncoderBlock(config.num_filters, config.num_filters, config.kernel_size, config.stride, config.padding)
                           for _ in range(config.num_layers-1)]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [DecoderBlock(config.num_filters, config.num_filters, config.kernel_size, config.stride, config.padding)
                          for _ in range(config.num_layers-1)]
        decoder_layers += [DecoderBlock(config.num_filters, config.num_channels, config.kernel_size, config.stride, config.padding,
                                        batch_norm=False, activation='tanh')]
        self.decoder = nn.Sequential(*decoder_layers)

        self.apply(weights_init)

    def forward(self, x):
        # need variance, concat the 3 channels
        x = self.encoder(x)
        x = self.decoder(x)
        return x

