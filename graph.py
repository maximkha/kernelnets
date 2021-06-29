import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(2)

import memlay

X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
#Y = torch.Tensor([0,1,1,0]).view(-1,1)
#Y = torch.Tensor([0,0,0,1]).view(-1,1)
#Y = torch.Tensor([1,1,1,0]).view(-1, 1)
Y = torch.Tensor([1,0,1,0]).view(-1, 1)

class XORNet(nn.Module):
    def __init__(self, input_dim = 2, output_dim=1):
        super(XORNet, self).__init__()
        #self.lin1 = nn.Linear(input_dim, 2)
        #self.lin2 = nn.Linear(2, output_dim)
        #self.lin1 = memlay.MemLayer(input_dim, output_dim)
        self.lin1 = memlay.MemLayer(input_dim, 2)
        self.lin2 = memlay.MemLayer(2, output_dim)

    def forward(self, x):
        x = self.lin1(x)
        #x = torch.sigmoid(x)
        x = self.lin2(x)
        return x

class XORLinNet(nn.Module):
    def __init__(self, input_dim = 2, output_dim=1):
        super(XORLinNet, self).__init__()
        self.lin1 = nn.Linear(input_dim, 1)
        self.lin2 = nn.Linear(2, output_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        # x = self.lin2(x)
        # x = torch.relu(x)
        return x

model = XORLinNet()

loss_func = nn.MSELoss()

# optimizer = optim.Adam(model.parameters(), lr=0.02)
optimizer = optim.Adam(model.parameters(), lr=0.002)

epochs = 1000#2001
steps = X.size(0)
for i in range(epochs):
    for j in range(steps):
        data_point = np.random.randint(X.size(0))
        x_var = Variable(X[data_point].unsqueeze(0), requires_grad=False)
        y_var = Variable(Y[data_point], requires_grad=False)
        
        optimizer.zero_grad()
        y_hat = model(x_var)
        loss = loss_func.forward(y_hat, y_var)
        loss.backward()
        optimizer.step()
        
    if i % 100 == 0 and i != 0:
        print(f"Epoch: {i}, Loss: {loss.data.numpy()}, ")

preds = model(X)
for x, y, y_h in zip(X, Y, preds):
    print(f"X = {x}, Y = {y}, Model(X) = {y_h}")
# for x, y in zip(X, Y):
#     print(f"X = {x}, Y = {y}, Model(X) = {model(x)}")

print(X)

res = 11 #2#101
Xs, Ys = np.meshgrid(np.linspace(0,1,res), np.linspace(0,1,res))
XY = np.array([Xs.flatten(), Ys.flatten()]).T
print(XY.shape)
print(XY)
Z = np.array(model(torch.Tensor(XY)).detach().numpy()).reshape(Xs.shape)
fig = plt.figure()
fig.suptitle("Model", fontsize=20)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xs, Ys, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Model(X,Y)=')
ax.view_init(45, 45)
plt.show()