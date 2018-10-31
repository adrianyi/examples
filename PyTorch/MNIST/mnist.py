from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter

from clusterone import get_data_path, get_logs_path

try:
    job_name = os.environ['JOB_NAME']
    task_index = int(os.environ['TASK_INDEX'])
    ps_hosts = os.environ['PS_HOSTS'].split(',')
    worker_hosts = os.environ['WORKER_HOSTS'].split(',')
    n_workers = len(worker_hosts)
    os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = worker_hosts[0].split(':')
except KeyError:
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None
    n_workers = 1
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'


def str2bool(v):
    """"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise IOError('Boolean value expected (i.e. yes/no, true/false, y/n, t/f, 1/0).')


def get_args():
    """Return parsed args"""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--local_data_dir', type=str, default='data/',
                        help='Path to local data directory')
    parser.add_argument('--local_log_dir', type=str, default='logs/',
                        help='Path to local log directory')
    parser.add_argument('--dist', type=str2bool, default='False')
    # Model params
    parser.add_argument('--hidden_units', type=int, nargs='*', default=[32, 64])
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=9999999)
    parser.add_argument('--cuda', type=str2bool, default=None,
                        help='Use CUDA. If left empty, CUDA will be used if available.')
    parser.add_argument('--ckpt_epochs', type=int, default=1)
    # Logging
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Number of steps before saving loss, etc.')
    parser.add_argument('--log_level', type=str, default='info',
                        choices=['info', 'debug'])
    opts = parser.parse_args()

    opts.data_dir = get_data_path(dataset_name = '*/*',
                                 local_root = opts.local_data_dir,
                                 local_repo = '',
                                 path = '')
    opts.log_dir = get_logs_path(root = opts.local_log_dir)

    opts.cuda = opts.cuda or torch.cuda.is_available()
    opts.device = torch.device('cuda' if opts.cuda else 'cpu')

    opts.distributed = n_workers > 1 or opts.dist

    return opts


class AttrProxy(object):
    """Borrowed from https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2"""
    def __init__(self, module, prefix, n_layers):
        self.module = module
        self.prefix = prefix
        self.length = n_layers

    def __getitem__(self, index):
        return getattr(self.module, self.prefix+str(index))

    def __iter__(self):
        return (getattr(self.module, self.prefix+str(i)) for i in range(self.length))


class CNNModel(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.dropout = opts.dropout
        units = [1] + opts.hidden_units
        for i in range(len(opts.hidden_units)):
            self.add_module('conv'+str(i), nn.Conv2d(units[i], units[i+1], kernel_size=3, stride=2))
        self.convs = AttrProxy(self, 'conv', len(opts.hidden_units))
        self.fc = nn.Linear(units[-1], 10)
        # self.writer = SummaryWriter(log_dir=opts.log_dir)
        self.train_steps = 0
        self.last_log_time = None

    # def add_graph_to_tensorboard(self):
    #     dummy_input = torch.autograd.Variable(torch.rand(1, 1, 28, 28)).to(self.opts.device)
    #     self.writer.add_graph(self, dummy_input)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.avg_pool2d(x, int(x.shape[2]))
        x = x.view(-1, self.opts.hidden_units[-1])
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def train_step(self, dataloader, optimizer, opts, epoch=-1):
        if self.last_log_time is None:
            self.last_log_time = time.time()
        self.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(opts.device), target.to(opts.device)
            optimizer.zero_grad()
            output = self.forward(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            self.train_steps += 1
            if self.train_steps % opts.log_freq == 0:
                seconds = time.time() - self.last_log_time
                steps_per_sec = float(opts.log_freq)/seconds
                # self.writer.add_scalars('loss/train',
                #                         {'train_loss': loss.item()},
                #                         self.train_steps)
                # self.writer.add_scalars('steps/',
                #                         {'steps_per_sec': steps_per_sec},
                #                         self.train_steps)

                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tSteps/s {:.3f}\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(dataloader.dataset),
                    100. * batch_idx / len(dataloader),
                    steps_per_sec,
                    loss.item()))
                self.last_log_time = time.time()

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
        # self.writer.add_scalars('loss/eval',
        #                         {'eval_loss': eval_loss},
        #                         self.train_steps)
        # self.writer.add_scalars('accuracy/eval',
        #                         {'eval_accuracy': accuracy},
        #                         self.train_steps)
        logger.info('Evaluation: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            eval_loss, correct, len(dataloader.dataset),
            100.*accuracy))


def main(opts):
    if opts.distributed:
        # dist.init_process_group(backend='mpi', rank=task_index, init_method=worker_hosts[0], world_size=n_workers)
        # '': '3362258944.0;tcp4://127.0.0.1:61761'
        os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = os.environ['PMIX_SERVER_URI2'].split('//')[1].split(':')
        # os.environ['MASTER_ADDR'] = '192.168.1.71'
        # os.environ['MASTER_PORT'] = '61587'
        dist.init_process_group(backend='mpi', rank=0, world_size=0)

    train_dataset = datasets.MNIST(opts.data_dir, train=True, download=True,
                                   transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    if opts.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=opts.cuda, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(opts.data_dir, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=opts.cuda)

    model = CNNModel(opts).to(opts.device)
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.-opts.learning_decay, last_epoch=-1)

    # model.add_graph_to_tensorboard()
    for epoch in range(1, opts.epochs + 1):
        if opts.distributed:
            train_sampler.set_epoch(epoch)
        scheduler.step()
        model.train_step(train_loader, optimizer, opts, epoch)
        model.evaluate(test_loader, opts, epoch)
        if epoch % opts.ckpt_epochs == 0:
            if not os.path.exists(opts.log_dir):
                os.makedirs(opts.log_dir)
            save_fp = os.path.join(opts.log_dir, 'saved_model.pt')
            logger.info('Saving checkpoint to {}'.format(save_fp))
            torch.save(model, save_fp)


if __name__ == '__main__':
    opts = get_args()
    logger = logging.getLogger()
    logging.basicConfig(level=logging.getLevelName(opts.log_level.upper()),
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S')
    
    logger.debug(os.environ)
    logger.debug(opts)
    main(opts)
