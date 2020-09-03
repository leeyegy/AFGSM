from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class AFGSM():
    '''
    Notice that if set ODI_num_size = 0, then FGSM_ODI acts as same as FGSM
    '''
    def __init__(self, model, max_val, min_val,loss,device,epsilon):
        self.model = model
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # loss function
        self.loss = loss
        # device | cpu or gpu
        self.device = device
        # if random start
        self.epsilon = epsilon

    def step_size_schedule(self,epoch):
        # 3 5 10
        # [1- 21] : step_size: 0.07/3 = 0.024
        # [22- 71] : step_size: 0.07/5 = 0.014
        # [72-]: step_size: 0.07/10 = 0.007
        if epoch>71:
            return 0.007
        elif epoch>21:
            return 0.014
        else:
            return 0.024

    def perturb(self,X,y,epoch,prev):
        step_size = self.step_size_schedule(epoch)
        # print("扰动步长:{}".format(step_size))
        # print("初始的扰动范围max:{}".format(torch.max(prev-X)))
        X_adv = Variable(prev.clone().detach().data, requires_grad=True)
        opt = optim.SGD([X_adv], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = self.loss(self.model(X_adv), y)
        loss.backward()
        eta = step_size * X_adv.grad.data.sign()

        X_adv = Variable(X_adv.data + eta, requires_grad=True)
        eta = torch.clamp(X_adv.data - X.data, -self.epsilon, self.epsilon)
        X_adv = Variable(X.data + eta, requires_grad=True)
        X_adv = Variable(torch.clamp(X_adv, self.min_val, self.max_val), requires_grad=True)
        return X_adv.clone().detach()

# decay version
    # def perturb(self,X,y,epoch,prev):
    #     # X_adv = Variable(prev.data, requires_grad=True)
    #     X_adv = Variable(X.data, requires_grad=True)
    #     # if self.random_start:
    #     #     random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)
    #     #     X_adv = Variable(X_adv.data + random_noise, requires_grad=True)
    #     loss_ratio = 1
    #
    #     rd = 0.5
    #     decay_factor = min(1,epoch/(rd*120))
    #     dropout = 1 - 0.7*decay_factor
    #
    #     for i in range(self.max_iter):
    #         opt = optim.SGD([X_adv], lr=1e-3)
    #         opt.zero_grad()
    #         with torch.enable_grad():
    #             loss = self.loss(self.model(X_adv), y)
    #             loss_ratio /= loss.item()
    #         print("cln loss:{}".format(loss.item()))
    #         # print("iter :{} loss:{}".format(i,loss))
    #         loss.backward()
    #         eta = self.PGD_step_size * X_adv.grad.data*self.factor
    #         # if i==0:
    #             # print(X_adv.grad.data)
    #         # print("梯度的最大值：{}，最小值：{}，平均值：{}".format(torch.max(X_adv.grad),torch.min(X_adv.grad),torch.mean(X_adv.grad)))
    #
    #         X_adv = Variable(X_adv.data + eta, requires_grad=True)
    #         eta = torch.clamp(X_adv.data - X.data, -self.epsilon, self.epsilon)
    #         X_adv = Variable(X.data + eta, requires_grad=True)
    #         X_adv = Variable(torch.clamp(X_adv, self.min_val, self.max_val), requires_grad=True)
    #
    #         eta = X_adv.data -X.data
    #         # print("扰动的最大值：{} 最小值：{}，平均值：{}".format(torch.max(eta),torch.min(eta),torch.mean(eta)))
    #         print("adv loss:{}".format(self.loss(self.model(X_adv),y).item()))
    #         loss_ratio *= self.loss(self.model(X_adv),y).item()
    #         # if loss_ratio <1:
    #         #     print("---------------")
    #         #     print("ALERT")
    #         #     print ("---------------")
    #         # print("loss ratio:{}".format(loss_ratio))
    #     # print(torch.max(prev))
    #     # print(dropout)
    #
    #     X_adv = Variable(X_adv.data*dropout+(1-dropout)*prev.data,requires_grad=True)
    #     X_adv = Variable(torch.clamp(X_adv, self.min_val, self.max_val), requires_grad=True)
    #
    #     return X_adv.detach()

