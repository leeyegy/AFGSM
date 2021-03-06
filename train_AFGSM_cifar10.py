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

def train(args, model, device, cifar_nat_x,cifar_x,cifar_y, optimizer, epoch):
    model.train()
    num_of_example = 50000
    batch_size = args.batch_size
    cur_order = np.random.permutation(num_of_example)
    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
    batch_idx = -batch_size

    for i in range(iter_num):
        # get shuffled data
        batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
        x_batch = cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].to(device)
        x_nat_batch = cifar_nat_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].to(device)
        y_batch = cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].to(device)

        batch_size = y_batch.shape[0]

        # attack
        with ctx_noparamgrad_and_eval(model):
            adv_data = adversary.perturb(x_nat_batch,y_batch,epoch,x_batch)


        # update cifar_x
        cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = adv_data.clone().detach().cpu()

        # loss backward
        optimizer.zero_grad()
        loss = F.cross_entropy(model(adv_data),y_batch,reduction='elementwise_mean')
        loss.backward()
        optimizer.step()

        # print max perturbation & min
        if i == 0:
            eta = adv_data - x_nat_batch
            print("max:{}".format(torch.max(eta)))
            print("min:{}".format(torch.min(eta)))

        # print progress
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(x_batch), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))

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
    prev = torch.ones([50000, 3, 32, 32])
    cifar_x, cifar_y = load_cifar10_data()
    cifar_nat_x = cifar_x.clone()
    # cifar_x = cifar_x.detach() + 0.001 * torch.randn(cifar_x.shape).detach() #random init

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # reset
        if epoch > 71:
            if (epoch-72)%10 == 0:
                cifar_x = cifar_nat_x.clone()
        elif epoch > 21:
            if (epoch-22)%5 == 0:
                cifar_x = cifar_nat_x.clone()
        else:
            if (epoch-1)%3 == 0:
                cifar_x = cifar_nat_x.clone()

        # adversarial training
        train(args, model, device, cifar_nat_x,cifar_x,cifar_y, optimizer, epoch)

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

def load_cifar10_data():
    data_ = torch.ones([50000, 3, 32, 32])
    target_ = torch.zeros([50000])
    for batch_idx,(data, target) in enumerate(train_loader):
        data_[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = data
        target_[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = target
    return data_,target_.long()

if __name__ == '__main__':
    main()
