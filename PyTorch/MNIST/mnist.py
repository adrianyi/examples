from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

from clusterone import get_data_path, get_logs_path

def str2bool(v):
    ''''''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise IOError('Boolean value expected (i.e. yes/no, true/false, y/n, t/f, 1/0).')

def get_args():
    '''Return parsed args'''
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--local_data_dir', type=str, default='data/',
                        help='Path to local data directory')
    parser.add_argument('--local_log_dir', type=str, default='logs/',
                        help='Path to local log directory')
    # Model params
    parser.add_argument('--hidden_units', type=int, nargs='*', default=[32, 64])
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=9999999)
    parser.add_argument('--cuda', type=str2bool, default=None,
                        help='Use CUDA. If left empty, CUDA will be used if available.')
    # Logging
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Number of steps before saving loss, etc.')
    opts = parser.parse_args()

    opts.data_dir = get_data_path(dataset_name = '*/*',
                                 local_root = opts.local_data_dir,
                                 local_repo = '',
                                 path = '')
    opts.log_dir = get_logs_path(root = opts.local_log_dir)

    if opts.cuda is None:
        opts.cuda = torch.cuda.is_available()
    opts.device = torch.device('cuda' if opts.cuda else 'cpu')

    return opts

class CNNModel(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.dropout = opts.dropout
        units = [1] + opts.hidden_units
        self.convs = [nn.Conv2d(units[i], units[i+1], 3, stride=2) for i in range(len(opts.hidden_units))]
        self.fc = nn.Linear(opts.hidden_units[-1], 10)
        self.writer = SummaryWriter(log_dir=opts.log_dir)

    def forward(self, x):
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x)), training=self.training)
        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(-1, self.opts.hidden_units[-1])
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def train_step(self, dataloader, optimizer, opts, epoch=-1):
        self.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(opts.device), target.to(opts.device)
            optimizer.zero_grad()
            output = self.forward(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if opts.i_iter % opts.log_freq == 0:
                seconds = time.time() - opts.last_log_time
                steps_per_sec = float(opts.log_freq)/seconds
                self.writer.add_scalars('loss/train',
                                        {'train_loss': loss.item()},
                                        opts.i_iter)
                self.writer.add_scalars('steps/',
                                        {'steps_per_sec': steps_per_sec},
                                        opts.i_iter)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tSteps/s {:.3f}\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(dataloader.dataset),
                    100. * batch_idx / len(dataloader),
                    steps_per_sec,
                    loss.item()))
                opts.last_log_time = time.time()
            opts.i_iter += 1

    def evaluate(self, dataloader, opts, epoch=-1):
        self.eval()
        eval_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(opts.device), target.to(opts.device)
                output = self.forward(data)
                eval_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        eval_loss /= len(dataloader.dataset)
        accuracy = float(correct) / len(dataloader.dataset)
        self.writer.add_scalars('loss/eval',
                                {'eval_loss': eval_loss},
                                opts.i_iter)
        self.writer.add_scalars('accuracy/eval',
                                {'eval_accuracy': accuracy},
                                opts.i_iter)
        print('Evaluation: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            eval_loss, correct, len(dataloader.dataset),
            100.*accuracy))

def main(opts):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(opts.data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=opts.cuda)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(opts.data_dir, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=opts.cuda)

    model = CNNModel(opts).to(opts.device)
    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.-opts.learning_decay, last_epoch=-1)

    opts.i_iter = 1
    opts.last_log_time = time.time()
    for epoch in range(1, opts.epochs + 1):
        scheduler.step()
        model.train_step(train_loader, optimizer, opts, epoch)
        model.evaluate(test_loader, opts, epoch)

if __name__ == '__main__':
    opts = get_args()
    print(opts)
    main(opts)
