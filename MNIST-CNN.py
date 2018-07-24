# import torch
# from torch.autograd import Variable

# tensor = torch.FloatTensor([[1, 2], [3, 4]])
# variable = Variable(tensor, requires_grad=True)

# t_out = torch.mean(tensor*tensor)
# v_out = torch.mean(variable*variable)

# v_out.backward()

# print(variable.data.numpy())

#----Realize one latent layer neural network-----#
# input dim = 3, latent layer = 4, output = 5
# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim

# x = Variable(torch.randn(4, 1), requires_grad=False)
# y = Variable(torch.randn(3, 1), requires_grad=False)

# w1 = Variable(torch.randn(5, 4), requires_grad=True)
# w2 = Variable(torch.randn(3, 5), requires_grad=True)

# def model_forward(x):
#     return F.sigmoid(w2 @ F.sigmoid(w1 @ x))

# optimizer = optim.SGD([w1, w2], lr=0.001)

# for epoch in range(10):
#     loss = nn.MSELoss(model_forward(x), y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


# #-----------predict curve------------#
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from torch.autograd import Variable

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict =torch.nn.Linear(n_hidden, n_output)

#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x

# net = Net(n_feature=1, n_hidden=10, n_output=1)

# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# y = x.pow(2) + 0.2*torch.rand(x.size())

# x, y = Variable(x), Variable(y)

# optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
# loss_func = torch.nn.MSELoss()

# plt.ion()   # 画图
# plt.show()

# for t in range(100):
#     prediction = prediction = net(x)
#     loss = loss_func(prediction, y)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if t % 5 == 0:
#         # plot and show learning process
#         plt.cla()
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), prediction.data.numpy(), lw=5)
#         plt.pause(0.1)

# import torch
# import torch.utils.data as Data

# BATCH_SIZE = 8

# x = torch.linspace(1, 10, 10)
# y = torch.linspace(10, 1, 10)

# torch_dataset = Data.TensorDataset(x, y)
# loader = Data.DataLoader(
#     dataset=torch_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=2,
# )

# for epoch in range(3):
#     for step, (batch_x, batch_y) in enumerate(loader):

#         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
#               batch_x.numpy(), '| batch y: ', batch_y.numpy())

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt
 
torch.manual_seed(1)    # reproducible
 
# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          
 
# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist',    
    train=True, 
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=False,
)

# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.show()

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
 
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:100]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:100]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,     
                out_channels=16,    
                kernel_size=5,      
                stride=1,          
                padding=2,      
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output
 
cnn = CNN()
#print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR) 
loss_func = nn.CrossEntropyLoss()  
 
# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y
 
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step%50==0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y.numpy()==test_y.numpy())/test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' %loss.data[0], '| Accuracy: %.4f' %accuracy)