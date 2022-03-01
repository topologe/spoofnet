import copy
from dataset import ImageDataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torchvision
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import argparse
import json

from AG2.model import AG2Config, AG2
from utils import save_images

KERNEL_SIZE = 21

NUM_GPU = 1 if torch.cuda.is_available() else 0


class AG2Trainer(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()

        self.args = args

        config = AG2Config(args)
        self.generator = AG2(config)
        self.resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
        for p in self.resnet.parameters():
            p.requires_grad = False

        self.save_hyperparameters()

    def forward(self, img):
        return self.generator(img)

    def training_step(self, batch, batch_idx):
        img, name, image_num, label = batch
        img = img.to(self.device)
        label = label.to(self.device)

        generated_gradient = self.generator(img)
        fake_image = generated_gradient + img
        label_loss, _ = self.get_label_loss(fake_image, label)

        if self.args['visibility_loss']:
            visibility_loss = self.get_visibility_loss(img, generated_gradient)
            loss = label_loss + self.args['visibility_loss_weight'] * visibility_loss

            self.log('visibility_loss', visibility_loss, on_step=True, on_epoch=False, prog_bar=False)
        else:
            loss = label_loss

        self.log('label_loss', label_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('total_loss', loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def configure_optimizers(self):
        opt1 = torch.optim.Adam(self.generator.parameters(),
                                lr=self.args['learning_rate'],
                                betas=(self.args['beta1'], self.args['beta2']))
        return [opt1]

    def get_label_loss(self, fake_image, label):
        self.resnet.eval()
        resnet_logits = self.resnet(fake_image)

        correct_logit = resnet_logits.gather(1, label.view(-1, 1))
        top_logits = torch.topk(resnet_logits, k=2).values
        min_logits = torch.cat([top_logits, correct_logit], dim=1).min(dim=1).values

        preds = resnet_logits.argmax(-1)

        loss = torch.mean(correct_logit.view(-1) - min_logits)
        return loss, preds

    def get_visibility_loss(self, img, gradient):
        """blur the image and compute the loss"""

        variance_loss_weight = 2

        img_var = self.get_pooled_variance(img, kernel_size=KERNEL_SIZE, stride=1)
        gradient_var = self.get_pooled_variance(gradient, kernel_size=KERNEL_SIZE, stride=1)

        # blur the squared gradient
        squared_gradient = (1 + gradient) ** 2
        #blurred_gradient = torch.nn.functional.avg_pool2d(squared_gradient, kernel_size=KERNEL_SIZE, stride=1,
        #                                             padding=int(KERNEL_SIZE / 2 - 0.5))
        blurred_gradient = torchvision.transforms.functional.gaussian_blur(squared_gradient, kernel_size=KERNEL_SIZE)

        # compute the loss
        loss =  2 * torch.pow(torch.mean(torch.abs(img_var - gradient_var)) + 1, 2)
        loss += 1 * torch.pow(torch.mean(torch.abs(gradient)) + 1, 2)
        loss += 2 * torch.mean(torch.pow(torch.abs(squared_gradient - blurred_gradient), 3))
        loss -= 3

        #loss += torch.mean(1 / (torch.abs(gradient) * img_var + 1e-12))
        #loss += torch.mean(torch.abs(gradient))
        #loss += 0.01 * gradient.view(len(gradient), -1).norm(dim=1).mean()
        return loss

    @staticmethod
    def get_pooled_variance(x, kernel_size=KERNEL_SIZE, stride=1):
        # use Var[x] = E[x^2] - E[x]^2
        padding = int(kernel_size / 2 - 0.5)
        e_x2 = torch.nn.functional.avg_pool2d(x ** 2, kernel_size=kernel_size, stride=stride, padding=padding)
        e_x_2 = torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding) ** 2
        return e_x2 - e_x_2

    def validation_step(self, batch, batch_idx):
        img, name, image_num, label = batch
        img = img.to(model.device)
        label = label.to(model.device)

        generated_gradient = model.generator(img)
        fake_image = generated_gradient + img
        label_loss, preds = model.get_label_loss(fake_image, label)

        accuracy = torch.sum(preds == label) / len(preds)
        self.log("val_loss", label_loss)
        self.log("val_acc", accuracy)


def evaluate(model, dataloader, log_dir):
    correct = 0.0
    total = 0.0
    for batch in tqdm(dataloader):
        img, name, image_num, label = batch
        img = img.to(model.device)
        label = label.to(model.device)

        generated_gradient = model.generator(img)
        fake_image = generated_gradient + img
        label_loss, preds = model.get_label_loss(fake_image, label)

        correct_array = preds == label
        correct += torch.sum(correct_array).item()
        total += len(preds)

        save_dir = log_dir + '/images'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        img_var = model.get_pooled_variance(img, kernel_size=KERNEL_SIZE, stride=1)
        grad_var = model.get_pooled_variance(generated_gradient, kernel_size=KERNEL_SIZE, stride=1)
        for i in range(len(img)):
            spoofed = 'no' if correct_array[i] else 'yes'
            fname = save_dir + f'/{name[i]}_{image_num[i]}_{spoofed}.jpg'
            save_images(img[i], generated_gradient[i], fake_image[i], img_var[i], grad_var[i], fname)

    acc = {'accuracy': f'{correct / total:.5f}'}
    print(acc)
    json.dump(acc, open(log_dir + '/accuracy.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_filters', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--visibility_loss', action='store_true')
    parser.add_argument('--visibility_loss_weight', type=int, default=10)
    parser.add_argument('--logdir', type=str, default="save/visibility_norm_loss")
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()
    #args = parser.parse_args(['--visibility_loss'])
    args = vars(args)
    args['logdir'] += f'/layers={args["num_layers"]}_kernel={args["kernel_size"]}_stride={args["stride"]}'
    print(args)

    dataset = ImageDataset(args['data_dir'])
    train_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)

    train_dataset.data, val_dataset.data = train_test_split(dataset.data, test_size=0.2)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)

    model = AG2Trainer(args)
    tb_logger = pl_loggers.TensorBoardLogger(args['logdir'])
    model_checkpoint = ModelCheckpoint(save_top_k=1)
    trainer = pl.Trainer(max_epochs=args['num_epochs'], gpus=NUM_GPU, log_every_n_steps=1, logger=tb_logger,
                         callbacks=[model_checkpoint])

    if args['pretrained'] is None:
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args['pretrained'])

    evaluate(model, val_dataloader, trainer.logger.log_dir)
