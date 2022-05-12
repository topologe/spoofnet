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
from PIL import Image

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
        # freeze all Resnet parameters
        for p in self.resnet.parameters():
            p.requires_grad = False

        self.save_hyperparameters()

    def forward(self, img):
        return self.generator(img)

    def training_step(self, batch, batch_idx):
        """Step computed on each batch during training"""
        # get batch attributes
        img, name, image_num, label = batch
        # move tensors to gpu if available
        img = img.to(self.device)
        label = label.to(self.device)

        # generate the gradient using AG2 generator
        generated_gradient = self.generator(img)
        # add gradient to the original image to create a fake image
        fake_image = generated_gradient + img
        # calculate the label loss
        label_loss, _ = self.get_label_loss(fake_image, label)

        if self.args['visibility_loss']:
            # calculate the visibility loss
            visibility_loss = self.get_visibility_loss(img, generated_gradient)
            # add scaled visibility loss to the label loss
            loss = label_loss + self.args['visibility_loss_weight'] * visibility_loss

            self.log('visibility_loss', visibility_loss, on_step=True, on_epoch=False, prog_bar=False)
        else:
            # if not using visibility loss, loss is simply the label loss
            loss = label_loss

        self.log('label_loss', label_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('total_loss', loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def configure_optimizers(self):
        """create the optimizer for PyTorch Lightning Trainer"""
        opt1 = torch.optim.Adam(self.generator.parameters(),
                                lr=self.args['learning_rate'],
                                betas=(self.args['beta1'], self.args['beta2']))
        return [opt1]

    def get_label_loss(self, fake_image, label):
        """Compute the label loss.
        Best case scenario: generator sussessfully confuses resnet, loss = 0
        Otherwise, loss is difference between the logit score of the truth and the second maximum logit score.
        """

        # make sure resnet is in eval mode
        self.resnet.eval()
        # compute the resnet logits for our fake image
        resnet_logits = self.resnet(fake_image)

        # for each item in the batch, gather the logit score for the truth label
        correct_logit = resnet_logits.gather(1, label.view(-1, 1))
        # get the top 2 logits from resnet
        top_logits = torch.topk(resnet_logits, k=2).values
        # concate the top 2 logits, with the correct logit score
        # take the minimum of these 3 values
        # Case 1: correct_logit == max_logit, then we have min([correct_logit, second_max_logit, correct_logit]) = second_max_logit
        # Case 2: correct_logit == second_max_logit, then we have min([max_logit, correct_logit, correct_logit]) = correct_logit
        # Case 3: second_max_logit > correct_logit, then we have min([max_logit, second_max_logit, correct_logit]) = correct_logit
        min_logits = torch.cat([top_logits, correct_logit], dim=1).min(dim=1).values

        preds = resnet_logits.argmax(-1)

        # calculate mean difference between correct logit and min_logit
        # if Case 1, loss = correct_logit - second_max_logit
        # if Case 2 or 3, loss = 0
        loss = torch.mean(correct_logit.view(-1) - min_logits)
        return loss, preds

    def get_visibility_loss(self, img, gradient):
        """
        Compute the visibility loss which measures the level of image noise that is detectable by humans.

        This loss has 3 components:
            Difference In Variance:         Computes the absolute difference between image and gradient variance.
                                            Intuitvely, we can insert a higher amount of noise into regions of the image
                                            which has high variance and still go undetected by humans.

            Magnitude of Gradient:          Measures the size of the gradient. We want to keep the gradient as small as
                                            possible so that a human does not know the image differs from the original.

            Magnitude of Gradient Variance: Measure the maximum size of the gradient variance. While we want to have
                                            gradient variance in high variance regions of the original image, we
                                            still want the variance as small as possible, while still confusing the
                                            model.
        """

        # first compute and store the blocked variance of the image and the gradient
        img_var = self.get_pooled_variance(img, kernel_size=KERNEL_SIZE, stride=1)
        gradient_var = self.get_pooled_variance(gradient, kernel_size=KERNEL_SIZE, stride=1)

        # Compute the loss

        # Difference In Variance
        loss = 1 * torch.pow(torch.mean(torch.abs(img_var - gradient_var)) + 1, 2)
        # Magnitude of Gradient
        loss += 1 * torch.pow(torch.mean(torch.abs(gradient)) + 1, 2)
        # Magnitude of Gradient Variance
        loss += 1 * torch.pow(torch.max(gradient_var) + 1, 2)
        # Substract 3 so that the loss as a minimum of 0.
        loss -= 3

        return loss

    @staticmethod
    def get_pooled_variance(x, kernel_size=KERNEL_SIZE, stride=1):
        """Calculates the pooled variance of an image"""
        # use Var[x] = E[x^2] - E[x]^2
        padding = int(kernel_size / 2 - 0.5)
        e_x2 = torch.nn.functional.avg_pool2d(x ** 2, kernel_size=kernel_size, stride=stride, padding=padding)
        e_x_2 = torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding) ** 2
        return e_x2 - e_x_2

    def validation_step(self, batch, batch_idx):
        """Step computed on each batch during validation"""
        # get batch attributes
        img, name, image_num, label = batch
        # move tensors to gpu if available
        img = img.to(model.device)
        label = label.to(model.device)

        # generate the gradient using AG2 generator
        generated_gradient = model.generator(img)
        # add gradient to the original image to create a fake image
        fake_image = generated_gradient + img
        # calculate the label loss only for logging
        label_loss, preds = model.get_label_loss(fake_image, label)

        # calculate resnet prediction accuracy on the fake images
        accuracy = torch.sum(preds == label) / len(preds)
        self.log("val_loss", label_loss)
        self.log("val_acc", accuracy)


def evaluate(args, model, dataloader, log_dir):
    """Evaluate the model."""

    correct = 0.0
    total = 0.0
    cache = {'img_var': [], 'gradient_var': [], 'gradient': []}
    for batch in tqdm(dataloader):
        # get batch attributes
        img, name, image_num, label = batch
        # move tensors to gpu if available
        img = img.to(model.device)
        label = label.to(model.device)

        # generate the gradient using AG2 generator
        generated_gradient = model.generator(img)
        # add gradient to the original image to create a fake image
        fake_image = generated_gradient + img
        # calculate the label loss only for logging
        label_loss, preds = model.get_label_loss(fake_image, label)

        # calculuate accuracy and store values
        correct_array = preds == label
        correct += torch.sum(correct_array).item()
        total += len(preds)

        save_dir = log_dir + '/images'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # save images along with gradients and variance images
        img_var = model.get_pooled_variance(img, kernel_size=KERNEL_SIZE, stride=1)
        grad_var = model.get_pooled_variance(generated_gradient, kernel_size=KERNEL_SIZE, stride=1)
        for i in range(len(img)):
            spoofed = 'no' if correct_array[i] else 'yes'
            fname = save_dir + f'/{name[i]}_{image_num[i]}_{spoofed}.jpg'
            original_image = Image.open(f'{args["data_dir"]}/lfw/{name[i]}/{name[i]}_{str(image_num[i].item()).zfill(4)}.jpg')
            save_images(img[i], generated_gradient[i], fake_image[i], img_var[i], grad_var[i], original_image, fname)

        cache['img_var'].append(img_var.detach().cpu())
        cache['gradient_var'].append(grad_var.detach().cpu())
        cache['gradient'].append(generated_gradient.detach().cpu())

    # print and save model accuracy
    acc = {'accuracy': f'{correct / total:.5f}'}
    print(acc)
    json.dump(acc, open(log_dir + '/accuracy.json', 'w'))

    # store variances and gradients for analysis
    cache['img_var'] = torch.cat(cache['img_var'], dim=0)
    cache['gradient_var'] = torch.cat(cache['gradient_var'], dim=0)
    cache['gradient'] = torch.cat(cache['gradient'], dim=0)
    torch.save(cache, log_dir + '/cache.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_filters', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
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
    args = vars(args)
    args['logdir'] += f'/layers={args["num_layers"]}_kernel={args["kernel_size"]}_stride={args["stride"]}'
    print(args)

    # create the dataset
    dataset = ImageDataset(args['data_dir'])
    train_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)

    # split the dataset into training and validation sets
    train_dataset.data, val_dataset.data = train_test_split(dataset.data, test_size=0.2)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)

    # initialize the model and PyTorch Lightning Trainer
    model = AG2Trainer(args)
    tb_logger = pl_loggers.TensorBoardLogger(args['logdir'])
    model_checkpoint = ModelCheckpoint(save_top_k=1)
    trainer = pl.Trainer(max_epochs=args['num_epochs'], gpus=NUM_GPU, log_every_n_steps=1, logger=tb_logger,
                         callbacks=[model_checkpoint])

    # fit the model on the training data
    if args['pretrained'] is None:
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args['pretrained'])

    # evaluate on the validation set
    evaluate(args, model, val_dataloader, trainer.logger.log_dir)
