from PIL import Image
import numpy as np


def save_images(img, gradient, fake_image, img_var, grad_var, fname):
    negative_gradient = -gradient
    images = [prep_image(x) for x in [img, img_var, gradient, negative_gradient, grad_var, fake_image]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(fname)


def prep_image(x):
    x = x.clip(0, 1) * 255
    x = x.permute(1, 2, 0)
    x = x.cpu().detach().numpy()
    x = x.round()
    x = x.astype(np.uint8)
    return Image.fromarray(x)
