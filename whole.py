import torch
from torch.autograd import Variable
from torch.autograd.grad_mode import F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(2)

import memlay

X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
Y = torch.Tensor([0,1,1,0]).view(-1,1) #xor
#Y = torch.Tensor([0,0,0,1]).view(-1,1) #and
#Y = torch.Tensor([1,1,1,0]).view(-1, 1) #weird
#Y = torch.Tensor([1,0,1,0]).view(-1, 1) #not second bit

class XORNet(nn.Module):
    def __init__(self, input_dim = 2, output_dim=1):
        super(XORNet, self).__init__()
        #self.lin1 = nn.Linear(input_dim, 1)
        #self.lin2 = nn.Linear(2, output_dim)
        self.lin1 = memlay.MemLayer(input_dim, 1, True)
        #self.lin3 = memlay.MemLayer(2, output_dim, True)
        #self.lin2 = memlay.MemLayer(2, output_dim, True)

    def forward(self, x):
        x = self.lin1(x)
        #x = torch.relu(x)
        #x = torch.sigmoid(x)
        
        #x = self.lin2(x)
        return x

model = XORNet()

loss_func = nn.MSELoss()

# optim.SGD(model.parameters(), lr=.002)
#optimizer = optim.Adam(model.parameters(), lr=0.02)
optimizer = optim.Adagrad(model.parameters(), lr=0.02)

loss_y = []
loss_x = []

epochs = 2001
steps = X.size(0)
for i in range(epochs):
    for j in range(steps):
        data_point = np.random.randint(X.size(0))
        x_var = Variable(X[data_point].unsqueeze(0), requires_grad=False)
        y_var = Variable(Y[data_point], requires_grad=False)

        optimizer.zero_grad()
        y_hat = model(x_var)
        # print(y_hat.shape)
        # print(y_hat.squeeze(0))
        # print(y_var.shape)
        # print(x_var)
        # exit()
        loss = loss_func.forward(y_hat.squeeze(0), y_var)
        loss.backward()
        optimizer.step()
        
    if i % 100 == 0 and i != 0:
        loss_y.append(loss.data.numpy())
        loss_x.append(i)
        print(f"Epoch: {i}, Loss: {loss.data.numpy()}, ")

# print(f"{model.lin1(X[0])}")
# print(f"{model.lin1(X)}")

preds = model(X)
for x, y, y_h in zip(X, Y, preds):
    print(f"X = {x}, Y = {y}, Model(X) = {y_h}")
# for x, y in zip(X, Y):
#     print(f"X = {x}, Y = {y}, Model(X) = {model(x)}")

from pytorch_model_summary import summary
print(summary(model, torch.zeros((1, 2)), show_input=False))

import matplotlib.pyplot as plt

plt.plot(loss_x, loss_y)
plt.show()