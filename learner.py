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
  """Calculates LTU threshold according to weights"""
  S_i = 0
  for weight in weights:
    if weight<0:
      S_i += 1
  threshold = len(weights)*0.6 - S_i
  return threshold

def ltu(input, weights):
  """LTU logic"""
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
  """LTU activation function"""
  def __init__(self, n_inp, n_tl1):
    super().__init__()
    self.weight = Parameter(torch.Tensor(n_tl1, n_inp))
  
  def forward(self, input):
    return ltu(input, self.weight)

def input_weight_init(inp, out):
  """Function for initializing input weight"""
  weight_choice = [1,-1]
  inp_weights = np.random.choice(weight_choice, (out,inp))
  return torch.from_numpy(inp_weights).float()
  
activation_function = LTU(n_inp, n_tl1)
torch.manual_seed(0)
tnet = nn.Sequential(nn.Linear(n_inp, n_tl1, bias=True), activation_function, nn.Linear(n_tl1, 1))

with torch.no_grad():
    #Input weights initialized with +1/-1
    tnet[0].weight.data = input_weight_init(n_inp, n_tl1)
    #Output layer weights initialized with N(0,1)
    torch.nn.init.normal_(tnet[2].weight, mean=0.0, std=1.0)
    tnet[1].weight = tnet[0].weight
lossfunc = nn.MSELoss()


for n_l1 in [10, 30, 100]:
    torch.manual_seed(1000)
    activation_function_ = LTU(n_inp, n_l1)
    net = nn.Sequential(nn.Linear(n_inp, n_l1), activation_function_, nn.Linear(n_l1, 1))
    with torch.no_grad():
        net[0].weight.data = input_weight_init(n_inp, n_l1)
        net[1].weight = net[0].weight 
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