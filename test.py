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

import torch
import torch.utils.data as Data

BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):

        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())