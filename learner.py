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
  S_i = len(weights[weights<0])
  threshold = len(weights)*0.6 - S_i
  return threshold

def ltu(input, weights):
  """LTU logic"""
  for i in range(len(input)):
    tau_i = calculate_threshold(weights[i])
    if input[i] < tau_i:
      input[i] = 0.0
    else: 
      input[i] = 1.0
  return input

class LTU(nn.Module):
  """LTU activation function"""
  def __init__(self, n_inp, n_tl1):
    super().__init__()
    self.weight = Parameter(torch.Tensor(n_tl1, n_inp))
    #To record output features of LTU layer
    self.out_features = None
  
  def forward(self, input):
    input = ltu(input, self.weight)
    self.out_features = input.clone()
    return input

def input_weight_init(inp, out):
  """Function for initializing input weight"""
  weight_choice = [1,-1]
  inp_weights = np.random.choice(weight_choice, (out,inp))
  return torch.from_numpy(inp_weights).float()

def update_lr(optimizer,lr):
    """Scheduler to update learning rate at every iteration"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

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


for n_l1 in [10, 30, 100, 1000]:
    torch.manual_seed(1000)
    activation_function_ = LTU(n_inp, n_l1)
    net = nn.Sequential(nn.Linear(n_inp, n_l1), activation_function_, nn.Linear(n_l1, 1))
    with torch.no_grad():
        net[0].weight.data = input_weight_init(n_inp, n_l1)
        net[1].weight = net[0].weight
        torch.nn.init.zeros_(net[2].weight)
        torch.nn.init.zeros_(net[2].bias)
    
    sgd = optim.SGD(net[2:].parameters(), lr = 0.0)
    torch.manual_seed(2000)
    losses = []
    sample_average = 0.0
    for t in range(T):
        inp = torch.rand(n_inp)
        target = tnet(inp) + torch.randn(1)
        pred = net(inp)
        loss = lossfunc(target, pred)
        losses.append(loss.item())
        net.zero_grad()
        loss.backward()
        #Evaluate step size parameter
        f_out = net[1].out_features
        sample_average = (sample_average *t + (f_out.norm()**2).item())/(t+1)
        step_size_param = 0.1/sample_average
        sgd = update_lr(sgd,step_size_param)
        sgd.step()
        
    losses = np.array(losses)
    bin_losses = losses.reshape(T//nbin, nbin).mean(1)
    plt.plot(range(0, T, nbin), bin_losses, label=n_l1)
plt.legend()
plt.show()