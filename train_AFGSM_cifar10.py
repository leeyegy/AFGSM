from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from advertorch.context import ctx_noparamgrad_and_eval
from models.wideresnet import *
from models.resnet import *
from Advanced_FGSM import *
from advertorch.attacks import LinfPGDAttack, GradientSignAttack

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-ResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')


parser.add_argument("--test_model_path",type=str)

#
parser.add_argument("--attack_method",default="AFGSM",choices=["PGD","FGSM","AFGSM"],type=str)


args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='/home/Leeyegy/.torch/datasets/', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='/home/Leeyegy/.torch/datasets/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# define net
model = ResNet18().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# define attacker
if args.attack_method == "AFGSM":
    adversary = AFGSM(model, epsilon=args.epsilon,
                     max_val=1.0, min_val=0.0, loss=nn.CrossEntropyLoss(), device=device)
elif args.attack_method=="FGSM":
    adversary = GradientSignAttack(
            model,loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            clip_min=0.0, clip_max=1.0,eps=args.epsilon,targeted=False)

PGD_adversary = LinfPGDAttack(model,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.03137,nb_iter=10,eps_iter=0.007,rand_init=True,clip_min=0.0,clip_max=1.0,targeted=False)
prev = torch.ones([50000,3,32,32])

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # reset prev
        if epoch > 71:
            if (epoch-72)%10 == 0:
                prev[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]=data.cpu()
        elif epoch > 21:
            if (epoch-22)%5 == 0:
                prev[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]=data.cpu()
        else:
            if (epoch-1)%3 == 0:
                prev[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]=data.cpu()
        optimizer.zero_grad()
        batch_prev = prev[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size].to(device)

        # attack
        with ctx_noparamgrad_and_eval(model):
            adv_data = adversary.perturb(data,target,epoch,batch_prev)
            prev[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = adv_data.cpu() # update prev

        loss = F.cross_entropy(model(adv_data),target,reduction='elementwise_mean')
        loss.backward()
        optimizer.step()

        # print max perturbation & min
        if batch_idx == 0:
            eta = adv_data - data
            print("max:{}".format(torch.max(eta)))
            print("min:{}".format(torch.min(eta)))

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    adv_correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        with ctx_noparamgrad_and_eval(model):
            adv_data = PGD_adversary.perturb(data,target)
        with torch.no_grad():
            output = model(adv_data)
        pred = output.max(1, keepdim=True)[1]
        adv_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Test Adv Accuracy: {}/{} ({:.0f}%)'.format(
        adv_correct, len(test_loader.dataset),
        100. * adv_correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-res18-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-res18-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
