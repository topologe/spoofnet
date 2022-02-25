import torch
from AG2.model import AG2Config, AG2

args = {'image_size': 160,
        'num_channels': 3,
        'num_filters': 32
        }

config = AG2Config(args)
generator = AG2(config)

img = torch.randn((1, args['num_channels'], args['image_size'], args['image_size']))

generated_gradient = generator(img)

