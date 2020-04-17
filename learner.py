import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from random import choice
from torch.nn.parameter import Parameter

T = 30000
nbin = 200
n_inp = 10
n_tl1 = 20
def calculate_threshold(weights):
  S_i = 0
  for weight in weights:
    if weight<0:
      S_i += 1
  threshold = len(weights)*0.6 - S_i
  return threshold

def ltu(input, weights):
  i = 0
  for i in range(len(input)):
    tau_i = calculate_threshold(weights[i])
    if input[i] < tau_i:
      input[i] = 0.0
    else: 
      input[i] = 1.0
    i=i+1
  return input

class LTU(nn.Module):
  def __init__(self, n_inp, n_tl1):
    super().__init__()
    self.weight = Parameter(torch.Tensor(n_tl1, n_inp))
  
  def forward(self, input):
    return ltu(input, self.weight)

def input_weight_init(size):
  weight_choice = [1.00,-1.00]
  inp_weights = np.random.choice(weight_choice, size)
  return torch.from_numpy(inp_weights)

activation_function = LTU(n_inp, n_tl1)
torch.manual_seed(0)
tnet = nn.Sequential(nn.Linear(n_inp, n_tl1, bias=True), activation_function, nn.Linear(n_tl1, 1))

with torch.no_grad():
    tnet[1].weight.data = tnet[0].weight.data
lossfunc = nn.MSELoss()

for n_l1 in [10, 30, 100, 1000]:
    torch.manual_seed(1000)
    net = nn.Sequential(nn.Linear(n_inp, n_l1), nn.Sigmoid(), nn.Linear(n_l1, 1))
    with torch.no_grad():
        net[2].weight *= 0
        net[2].bias *= 0
    sgd = optim.SGD(net[2:].parameters(), lr=1/n_l1)

    torch.manual_seed(2000)
    losses = []
    for t in range(T):
        inp = torch.rand(n_inp)
        target = tnet(inp)+ torch.randn(1)
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