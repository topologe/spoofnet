from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np


def save_images(img, gradient, fake_image, img_var, grad_var, original_image, fname):
    images = [prep_image(img),
              prep_image(img_var, 'var'),
              prep_image(gradient, 'grad'),
              prep_image(-gradient, 'grad'),
              prep_image(grad_var, 'var'),
              prep_image(fake_image),
              original_image,
              reproject_face(original_image, fake_image)
              ]

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(fname)


def prep_image(x, mode=None):
    if mode == 'grad':
        x -= 1
        x = x.clip(min=-1)
    elif mode == 'var':
        x -= 1

    x = x * 128 + 127.5
    x = x.permute(1, 2, 0)
    x = x.cpu().detach().numpy()
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def reproject_face(original_image, generated_image):
    mtcnn = MTCNN()
    boxes, probs, points = mtcnn.detect(original_image, landmarks=True)
    box, probs, points = mtcnn.select_boxes(boxes, probs, points, original_image, method=mtcnn.selection_method)

    box = box[0]
    margin = [mtcnn.margin * (box[2] - box[0]) / (mtcnn.image_size - mtcnn.margin),
              mtcnn.margin * (box[3] - box[1]) / (mtcnn.image_size - mtcnn.margin)]
    raw_image_size = original_image.size
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    w, h = box[3] - box[1], box[2] - box[0]

    new = Image.new('RGB', (160, 160))
    new.paste(prep_image(generated_image), (0, 0))

    out = new.copy().resize((h, w), Image.BILINEAR)

    projected_image = original_image.copy()
    projected_image.paste(out, (box[0], box[1]))
    return projected_image
