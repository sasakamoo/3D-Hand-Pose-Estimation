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

            #with open(os.path.join(self.root, "FreiHAND_pub_v2_eval/evaluation_verts.json"), 'r') as f:
            #    self.verts = json.load(f)

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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.eval:
            img_name = os.path.join(self.root, "FreiHAND_pub_v2_eval/evaluation/rgb/", f"{idx:0{8}d}.jpg")
        else:
            img_name = os.path.join(self.root, "FreiHAND_pub_v2/training/rgb/", f"{idx:0{8}d}.jpg")

        feature = io.decode_image(img_name).to(torch.float32)
        points = torch.tensor(self.points[idx])
        K = torch.tensor(self.K[idx])
        scale = torch.tensor(self.scale[idx])
        label = {'Points': points, 'K': K, 'Scale': scale}
        return feature, label

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target['Points'].to(device)
        optimizer.zero_grad()
        output = model(data)
        t = target.size()
        s = output.size()
        loss = F.mse_loss(output, target, reduction='sum')
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
            data, target = data.to(device), target['Points'].to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def projectPoints(xyz, K):     
    p = torch.matmul(K, xyz.t()).t()  
    return p[:, :2] / p[:, -1:]

def plot_hand(points):
    cmap = {
        "thumb": {"idx": [0, 1, 2, 3, 4], "color": 'r'},
        "index": {"idx": [0, 5, 6, 7, 8], "color": 'm'},
        "middle": {"idx": [0, 9, 10, 11, 12], "color": 'b'},
        "ring": {"idx": [0, 13, 14, 15, 16], "color": 'c'},
        "little": {"idx": [0, 17, 18, 19, 20], "color": 'g'}
    }

    for finger, params in cmap.items():
        plt.plot(points[params["idx"], 0], points[params["idx"], 1], params["color"])

class ConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_depth),
            nn.Conv2d(in_depth, out_depth, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_depth),
            nn.Conv2d(out_depth, out_depth, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super().__init__()
        self.convblock1 = ConvBlock(in_channel, N)
        self.convblock2 = ConvBlock(N, 2*N)
        self.convblock3 = ConvBlock(2*N, 4*N)
        self.convblock4 = ConvBlock(4*N, 8*N)

        self.convblock5 = ConvBlock(4*N + 8*N, 4*N)
        self.convblock6 = ConvBlock(2*N + 4*N, 2*N)
        self.convblock7 = ConvBlock(N + 2*N, N)

        self.convoutput = nn.Sequential(
            nn.Conv2d(N, out_channel, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.flat = nn.Flatten(2, -1)
        self.output = nn.Linear(224*224, 3)
        
    def forward(self, x):
        conv_x1 = self.convblock1(x)
        conv_x2 = self.convblock2(self.maxpool(conv_x1))
        conv_x3 = self.convblock3(self.maxpool(conv_x2))
        conv_x4 = self.convblock4(self.maxpool(conv_x3))

        conv_u1 = self.convblock5(torch.cat([self.upsample(conv_x4), conv_x3], dim=1))
        conv_u2 = self.convblock6(torch.cat([self.upsample(conv_u1), conv_x2], dim=1))
        conv_u3 = self.convblock7(torch.cat([self.upsample(conv_u2), conv_x1], dim=1))
        conv_l = self.convoutput(conv_u3)
        flat_l = self.flat(conv_l)
        return self.output(flat_l)
        #return self.convoutput(conv_u3) # This is for training against heatmaps

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
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


    #features, labels = next(iter(training_loader))
    #for i in range (0, len(features)):
    #    feature = features[i]
    #    points = labels['Points'][i]
    #    K = labels['K'][i]
    #    scale = labels['Scale'][i]
    #    imgPoints = projectPoints(points, K) 
    #    img = feature.detach().numpy()
    #    img = img.transpose((1, 2, 0))
    #    plt.imshow(img)
    #    plot_hand(imgPoints)
    #    plt.show()
    #return
    model = UNet(3, 21, 16).to(device)
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