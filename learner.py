import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

T = 30000
nbin = 200
n_inp = 10
n_tl1 = 20
def ltu(input):
  i = 0
  for element in input:
    if input[i]<0.6:
      input[i]=0.0
    else: 
      input[i]=1.0
    i=i+1
  return input

class LTU(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return ltu(input)
  
  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad_input = torch.sigmoid(input-0.6)*(1-torch.sigmoid(input-0.6))
    return grad_input

class LTUact(nn.Module):
  def __init__(self):
    super(LTUact, self).__init__()
    self.act = LTU.apply
  
  def forward(self, x):
    x= self.act(x)
    return x

activation_function = LTUact()
torch.manual_seed(0)
tnet = nn.Sequential(nn.Linear(n_inp, n_tl1, bias=True), activation_function, nn.Linear(n_tl1, 1))
with torch.no_grad():
    tnet[2].weight *= 100
    tnet[0].weight *= 1
lossfunc = nn.MSELoss()

for n_l1 in [10, 30, 100, 1000]:
    torch.manual_seed(1000)
    net = nn.Sequential(nn.Linear(n_inp, n_l1), activation_function, nn.Linear(n_l1, 1))
    with torch.no_grad():
        net[2].weight *= 0
        net[2].bias *= 0
    sgd = optim.SGD(net[2:].parameters(), lr=1/n_l1)

    torch.manual_seed(2000)
    losses = []
    for t in range(T):
        inp = torch.rand(n_inp)
        target = tnet(inp) + torch.randn(1)
        pred = net(inp)
        loss = lossfunc(target, pred)
        losses.append(loss.item())
        net.zero_grad()
        loss.backward()
        sgd.step()
    losses = np.array(losses)
    bin_losses = losses.reshape(T//nbin, nbin).mean(1)
    plt.plot(range(0, T, nbin), bin_losses, label=n_l1)
plt.legend()
plt.show()