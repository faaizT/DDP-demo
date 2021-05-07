import torch
import torch.nn as nn
import torchvision
import torch.multiprocessing as mp
import torch.distributed as dist
from argparse import ArgumentParser
import os

def cleanup():
    dist.destroy_process_group()

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
    rank = gpu
    print("GPU", gpu)
    # rank calculation for each process per gpu so that they can be
    # identified uniquely.
    print('rank:',rank)
    # Boilerplate code to initialise the parallel process.
    # It looks for ip-address and port which we have set as environ variable.
    # If you don't want to set it in the main then you can pass it by replacing
    # the init_method as ='tcp://<ip-address>:<port>' after the backend.
    # More useful information can be found in
    # https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        rank=rank
    )
    torch.manual_seed(0)
    # start from the same randomness in different nodes.
    # If you don't set it then networks can have different weights in different
    # nodes when the training starts. We want exact copy of same network in all
    # the nodes. Then it will progress form there.

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./", train=True, transform=transform, download=True
    )
    # Ensures that each process gets differnt data from the batch.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # calculate the batch size for each process in the node.
        batch_size=int(128),
        shuffle=(train_sampler is None),
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )


    # load the model to the specified device, gpu-0 in our case
    model = AE(input_shape=784)
    model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=False
    )
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

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch: {}/{}, loss = {:.6f}".format(epoch+1, args.epochs, loss))
        if rank == 0:
            dict_model = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': args.epochs,
            }
            torch.save(dict_model, './model.pth')
    cleanup()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--local_ranks', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    parser.add_argument('--ip_address', type=str, required=True,
                        help='ip address of the host node')
    parser.add_argument("--checkpoint", default=None,
                        help="path to checkpoint to restore")
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    args.world_size = args.nodes
    # add the ip address to the environment variable so it can be easily avialbale
    os.environ['MASTER_ADDR'] = args.ip_address
    print("ip_adress is", args.ip_address)
    import socket
    print("Running on", socket.gethostname())

    os.environ['MASTER_PORT'] = '6969'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    # nprocs: number of process which is equal to args.ngpu here
    mp.spawn(train, nprocs=args.world_size, args=[args])