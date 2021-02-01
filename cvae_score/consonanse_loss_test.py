from __future__ import print_function
import argparse
import torch
import torch.utils.data
import random as rand
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from chord import chord
from dataset_test import Test_dataset

parser = argparse.ArgumentParser(description='VAE MUSIC SCORE ')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

td = Test_dataset()
train_loader = torch.utils.data.DataLoader(
    td, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    td, batch_size=1, shuffle=True)

device = torch.device("cuda" if args.cuda else "cpu")


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.latent_dim = 10
        self.input_dim = 16*((12 + 1)*2)
        self.fc1 = nn.Linear(self.input_dim + 1, 400)
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim + 1, 400)
        self.fc4 = nn.Linear(400, self.input_dim)

    def encode(self, x, labels):
        #print(x.view(-1, self.input_dim))
        h1 = F.relu(
            self.fc1(torch.cat((x.view(-1, self.input_dim), labels), dim=-1)))
        return self.fc21(h1)

    def decode(self, latent, consonanse, labels):
        h3 = F.relu(
            self.fc3(torch.cat((latent, consonanse.float()), dim=-1)))
        h4 = self.fc4(h3)
        recon_x = torch.t(h4).reshape(-1, (12 + 1) * 2)
        h_consonanse = self.calc_chord(recon_x, labels)
        return torch.sigmoid(h4), torch.sigmoid(h_consonanse)

    def forward(self, x, labels):
        x = x.float()
        latent = self.encode(x, self.calc_chord(x[0], labels))
        return self.decode(latent, self.calc_chord(x[0], labels), labels)

    def calc_chord(self, x, labels):
        cd = chord()
        return cd.test_calc(x, labels)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, consonanse, labels):
    cd = chord()
    BCE = F.binary_cross_entropy(
        recon_x, x.float().view(-1, 16*((12 + 1)*2)), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon_x = torch.t(recon_x).reshape(-1, (12+1)*2)
    # print(torch.t(x).reshape(-1, 13))

    """ConsonanseLoss = F.binary_cross_entropy(cd.loss_exam(recon_x, labels),
                                            cd.loss_exam(x[0], labels).float().detach())"""

    ConsonanseLoss = BCE.item()/10*(cd.loss_exam(recon_x, labels) -
                                    cd.loss_exam(x[0], labels).float().detach())**2

    # print(ConsonanseLoss)

    return BCE + ConsonanseLoss, ConsonanseLoss


def train(epoch):
    model.train()
    train_loss = 0
    train_consonanse_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # print(data.size(), labels.size())
        # print(labels)
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        recon_batch, consonanse = model(data, labels)
        loss, consonanse_loss = loss_function(
            recon_batch, data, consonanse, labels)
        loss.backward()
        train_loss += loss.item()
        train_consonanse_loss += consonanse_loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 10:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss, train_consonanse_loss


def out_test():
    model.eval()
    test_loss = 0
    label_flag = True
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(train_loader):
            # print(data.size(), labels.size())
            # print(labels)
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            recon_batch, consonanse = model(data, labels)
            recon_batch = torch.t(recon_batch).reshape(-1, (12 + 1) * 2)
            recon_batch = recon_batch.to('cpu').detach().numpy().copy()
            output = pd.DataFrame(
                recon_batch,
                columns=['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'rest',
                         'c_h', 'c#_h', 'd_h', 'd#_h', 'e_h', 'f_h', 'f#_h', 'g_h', 'g#_h',
                         'a_h', 'a#_h', 'b_h', 'rest_h'])
            print(output)
            output.to_csv('test_out.csv')
            break


if __name__ == "__main__":
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    loss_array = np.ndarray(0)
    consonanse_array = np.ndarray(0)
    epoch_array = np.arange(1, args.epochs+1)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_consonanse_loss = train(epoch)
        loss_array = np.append(loss_array, train_loss)
        consonanse_array = np.append(consonanse_array, train_consonanse_loss)
        # test(epoch)
    out_test()
    ax1.plot(epoch_array, loss_array, color="blue", label="loss")
    ax2.plot(epoch_array, consonanse_array,
             color="green", label="consonanse_loss")
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    plt.show()
