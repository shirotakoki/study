from __future__ import print_function
import argparse
import torch
import torch.utils.data
import random as rand
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image



parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.latent_dim = 2
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc22 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, labels):
        h1 = F.relu(self.fc1(torch.cat((x.view(-1, 784),labels.float().view(-1,10)),dim=-1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, labels):
        h3 = F.relu(self.fc3(torch.cat((z,labels.float().view(-1,10)),dim=-1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        labels = torch.eye(10)[labels]
        #print(data.size(), labels.size())
        #print(labels)
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    label_flag = False
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            if label_flag :
                labels = torch.eye(10)[labels]
                labels = labels.to(device)
            else:
                labels = np.zeros((labels.size()[0], 10))
                labels = torch.from_numpy(labels).clone()
                labels = labels.to(device)
            #print(labels.size())
            recon_batch, mu, logvar = model(data,labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'cvae/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, model.latent_dim).to(device)
            #sample_label = torch.eye(10)[torch.tensor([rand.randint(0, 9)])]
            sample_label = np.zeros((64, 10))
            """
            for i in range(64):
                sample_label[i][rand.randint(0, 9)] = 1
            """
            sample_label = torch.from_numpy(sample_label).clone()
                
            #print(sample_label)
            sample = model.decode(sample, sample_label).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'cvae/results/sample_' + str(epoch) + '.png')

    sample_num = 2
    cmap = plt.get_cmap("tab10")
    for i, (data, labels) in enumerate(test_loader):
        one_hot_labels = torch.eye(10)[labels]
        data = data.to(device)
        one_hot_labels = one_hot_labels.to(device)
        recon_batch, mu, logvar = model(data, one_hot_labels)
        z = model.reparameterize(mu, logvar)
        z = z.to('cpu').detach().numpy().copy()
        labels = labels.to('cpu').detach().numpy().copy()
        for points,label in zip(z,labels):
            #print(points, label)
            plt.scatter(points[0],points[1],color=cmap(label))
        if sample_num == i: break
    plt.show()
        
