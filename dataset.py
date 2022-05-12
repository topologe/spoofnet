import torch
import os
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image

IMG_SOURCE = 'data/lfw'
IMG_SAVE = 'data/cropped_img'


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        save_fname = data_path + '/ag2_dataset.pt'
        if os.path.exists(save_fname):
            # load existing dataset if exists
            self.data = torch.load(save_fname)
        else:
            # create new dataset for training
            # load the labels used in the Facenet-vggface2 Resnet model
            labels = pd.read_csv('data/lfw/labels-vggface2.csv')
            # load list of images resnet predicts correctly. We want to ensure we only training on these images.
            df = pd.read_csv('utils/lfw_correct_images.txt')
            name_to_num = {v: k for k, v in labels.Name.to_dict().items()}
            df['label'] = df.name.map(name_to_num)
            df['path'] = df.apply(lambda x: f'{IMG_SOURCE}/{x["name"]}/{x["name"]}_{str(x.image_num).zfill(4)}.jpg', axis=1)
            df['save_path'] = df['path'].apply(lambda x: f'{IMG_SAVE}/{x.split("/")[-1]}')

            self.data = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                img = Image.open(row.path)
                mtcnn = MTCNN()
                img = mtcnn(img, row.save_path)
                self.data.append({'img': img,
                                  'name': row['name'],
                                  'image_num': row.image_num,
                                  'label': row['label']})

            torch.save(self.data, save_fname)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]['img'], \
               self.data[item]['name'], \
               self.data[item]['image_num'], \
               torch.tensor(self.data[item]['label'])


