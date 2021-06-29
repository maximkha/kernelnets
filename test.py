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

lay = memlay.MemLayer(1, 1, False)

# print(lay.bias)
print(lay.weights)

# print("=====")
# testdata = torch.tensor([[30.]], dtype=torch.float, requires_grad=True)
# print(testdata)
# outp = lay.forward(testdata)
# outp.retain_grad()
# print(outp)
# print(outp.backward())
# print(outp.grad)

print("=====")
testdata = torch.tensor([[100.]], dtype=torch.float, requires_grad=True)
print(testdata)
outp = lay.forward(testdata)
outp.retain_grad()
print(outp)
#print(outp.backward())
#print("grad")
#print(torch.autograd.grad(outp, testdata, create_graph=True)[0])
print("grad2")
#print(torch.autograd.grad(outp, lay.parameters(), create_graph=True, allow_unused=True))
target = torch.tensor(1)
loss = (outp - target) ** 2
loss.backward()

print(outp.grad)
print(list(lay.parameters()))