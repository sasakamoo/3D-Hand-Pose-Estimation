import argparse
import json
import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, io
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np

class FreiHAND_Dataset(Dataset):
    def __init__(self, root, eval=False):
        """
        Arguments: 
            eval (bool): Load evaluation set when true else load training set
            root (string): Root directory of FreiHAND Dataset
        """
        self.root = root
        self.eval = eval
        
        if eval:
            with open(os.path.join(self.root, "FreiHAND_pub_v2/training_K.json"), 'r') as f:
                self.K = json.load(f)

            with open(os.path.join(self.root, "FreiHAND_pub_v2_eval/evaluation_scale.json"), 'r') as f:
                self.scale = json.load(f)

            with open(os.path.join(self.root, "FreiHAND_pub_v2_eval/evaluation_verts.json"), 'r') as f:
                self.verts = json.load(f)

            with open(os.path.join(self.root, "FreiHAND_pub_v2_eval/evaluation_xyz.json"), 'r') as f:
                self.points = json.load(f)
        else: 
            with open(os.path.join(self.root, "FreiHAND_pub_v2/training_K.json"), 'r') as f:
                self.K = json.load(f)

            with open(os.path.join(self.root, "FreiHAND_pub_v2/training_scale.json"), 'r') as f:
                self.scale = json.load(f)

            # Verts file is huge for some reason
            #with open(os.path.join(self.root, "FreiHAND_pub_v2/training_verts.json"), 'r') as f:
            #    self.verts = json.load(f)

            with open(os.path.join(self.root, "FreiHAND_pub_v2/training_xyz.json"), 'r') as f:
                self.points = json.load(f)

    def __len__(self):
        return len(self.K)

    # TODO: FIX 
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.eval:
            img_name = os.path.join(self.root, "FreiHAND_pub_v2_eval/evaluation/rgb/", f"{idx:0{8}d}.jpg")
        else:
            img_name = os.path.join(self.root, "FreiHAND_pub_v2/training/rgb/", f"{idx:0{8}d}.jpg")

        img = io.read_image(img_name)

        return img

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)
    
    training_dataset = FreiHAND_Dataset("../Datasets")
    testing_dataset = FreiHAND_Dataset("../Datasets")
    training_loader = torch.utils.data.DataLoader(training_dataset,**train_kwargs)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, **test_kwargs)

    feature_vectors = next(iter(training_loader))
    img = feature_vectors[0]
    img_np = img.detach().numpy()
    img_np = img_np.transpose((1, 2, 0))
    plt.imshow(img_np)
    plt.show()
    return
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, training_loader, optimizer, epoch)
        test(model, device, testing_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()