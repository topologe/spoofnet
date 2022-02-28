from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import pandas as pd
import glob
from tqdm import tqdm

lfw_names = pd.read_table('data/lfw-names.txt', header=None, index_col=0, names=['Name', 'Count'])
lfw_names = lfw_names.loc[lfw_names.Count > 1, 'Count']
labels = pd.read_csv('data/lfw/labels-vggface2.csv')
lfw_names = lfw_names[lfw_names.index.isin(labels.Name)]
labels = labels.Name.to_dict()
name_to_num = {v: k for k, v in labels.items()}
names = lfw_names.index.tolist()

IMG_SOURCE = 'data/lfw'
IMG_SAVE = 'data/cropped_img'

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()

output = {'name': [], 'image_num': [], 'label': [], 'pred': []}
for name in tqdm(names):
    imgs = sorted(glob.glob(f'{IMG_SOURCE}/{name}/*.jpg'))
    inputs = []
    for path in imgs:
        fname = path.split("/")[-1]
        _, num = fname.split('.')[0].rsplit('_', 1)
        output['name'].append(name)
        output['image_num'].append(num)

        save_path = f'{IMG_SAVE}/{fname}'

        img = Image.open(path)
        img_cropped = mtcnn(img, save_path).unsqueeze(0)
        inputs.append(img_cropped)

    inputs = torch.cat(inputs)

    logits = resnet(inputs)
    preds = logits.argmax(-1)

    label = name_to_num[name]
    output['label'].extend([label]*len(imgs))
    output['pred'].extend(preds.tolist())

output = pd.DataFrame(output)
output.to_csv('utils/facenet_predictions.csv', index=False)

output.loc[output.label == output.pred, ['name', 'image_num']].to_csv('utils/lfw_correct_images.txt', index=False)