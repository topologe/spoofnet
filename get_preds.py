import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

from AG2.model import AG2, AG2Config
from utils import reproject_face

checkpoint = torch.load('save/visibility_layers=4_kernel=9_stride=1/checkpoints/epoch=3999-step=227999.ckpt',
                        map_location='cpu')

args = checkpoint['hyper_parameters']['args']
config = AG2Config(args)
model = AG2(config)

state_dict = {key.replace('generator.', ''): value for key, value in checkpoint['state_dict'].items() if 'generator' in key}
model.load_state_dict(state_dict)
model.eval()

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()

img_dir = 'data/lfw/'

image_files = [
    'Ashton_Kutcher/Ashton_Kutcher_0002.jpg',
    'Carrie-Anne_Moss/Carrie-Anne_Moss_0005.jpg',
    'Bill_Belichick/Bill_Belichick_0002.jpg',
    'Jon_Voight/Jon_Voight_0003.jpg',
    'Arianna_Huffington/Arianna_Huffington_0004.jpg',
    'Brad_Garrett/Brad_Garrett_0003.jpg',
    'Claire_Danes/Claire_Danes_0001.jpg',
    'David_Beckham/David_Beckham_0008.jpg',
    'John_Cusack/John_Cusack_0001.jpg',
    'Naomi_Watts/Naomi_Watts_0004.jpg',
    'Bob_Huggins/Bob_Huggins_0004.jpg',
    'Diana_Krall/Diana_Krall_0004.jpg',
    'Halle_Berry/Halle_Berry_0004.jpg',
    'Paul_McCartney/Paul_McCartney_0001.jpg',
    'Vicente_Fox/Vicente_Fox_0023.jpg',
    'Jennifer_Capriati/Jennifer_Capriati_0002.jpg',
    'Rio_Ferdinand/Rio_Ferdinand_0001.jpg',
    'Robert_De_Niro/Robert_De_Niro_0004.jpg',
    'Sophia_Loren/Sophia_Loren_0003.jpg',
    'Tiger_Woods/Tiger_Woods_0018.jpg'
]

labels = pd.read_csv('data/lfw/labels-vggface2.csv')

save_dir = 'best_images/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

output = []
gradients = []
for image_file in tqdm(image_files[::-1]):
    img = Image.open(img_dir + image_file)
    cropped_img = mtcnn(img)

    gradient = model(cropped_img.unsqueeze(0))[0]
    cropped_fake_img = cropped_img + gradient
    gradients.append(gradient)

    fake_img = reproject_face(img, cropped_fake_img)

    original_logits = resnet(cropped_img.unsqueeze(0))[0]
    original_probs = torch.softmax(original_logits, dim=0)

    fake_logits = resnet(cropped_fake_img.unsqueeze(0))[0]
    fake_probs = torch.softmax(fake_logits, dim=0)

    data = {'original_pred': labels.loc[original_probs.argmax().item(), 'Name'],
            'original_prob': original_probs.max().item(),
            'fake_pred': labels.loc[fake_probs.argmax().item(), 'Name'],
            'fake_prob': fake_probs.max().item()
            }

    #save_file = image_file.split('/', 1)[-1]
    #img.save(save_dir + save_file)
    #fake_img.save(save_dir + 'fake_' + save_file)

    output.append(data)


df = pd.DataFrame(output)
df.to_csv('best_images/results.csv')
