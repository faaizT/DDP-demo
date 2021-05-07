import torch
import torch.nn as nn
import torchvision
from argparse import ArgumentParser

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=kwargs["input_shape"], out_features=128),
            nn.ReLU(inplace=True),
            # small dimension
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            # Recconstruction of input
            nn.Linear(in_features=128, out_features=kwargs["input_shape"]),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        reconstructed = self.net(features)
        return reconstructed


def train(gpu, args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./", train=True, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4,
        pin_memory=True
    )

    # load the model to the specified device, gpu-0 in our case
    model = AE(input_shape=784)
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Loss function
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()
            # pe-rform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

            # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch: {}/{}, loss = {:.6f}".format(epoch+1, args.epochs, loss))


def main():
    parser = ArgumentParser()
    parser.add_argument('--ngpus', default=1, type=int,
                        help='number of gpus per node')

    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    train(0, args)

if __name__ == '__main__':
    main()