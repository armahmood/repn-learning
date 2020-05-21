import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from random import choice
from torch.nn.parameter import Parameter
import progressbar
import argparse
import os
import sys

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

def initialize_target_net(n_inp, n_tl1, seed_num_target):
  """Initializes target network"""
  torch.manual_seed(seed_num_target)
  activation_function = LTU(n_inp, n_tl1)
  tnet = nn.Sequential(nn.Linear(n_inp, n_tl1, bias=True), activation_function, nn.Linear(n_tl1, 1))
  with torch.no_grad():
    #Input weights initialized with +1/-1
    np.random.seed(seed_num_target)
    tnet[0].weight.data = input_weight_init(n_inp, n_tl1)
    #Output layer weights initialized with N(0,1)
    torch.nn.init.normal_(tnet[2].weight, mean=0.0, std=1.0)
    tnet[1].weight = tnet[0].weight
  return tnet

def initialize_learning_net(n_inp, n_l1, seed_num):
  """Initializes learning network"""
  torch.manual_seed(seed_num)
  activation_function_ = LTU(n_inp, n_l1)
  net = nn.Sequential(nn.Linear(n_inp, n_l1), activation_function_, nn.Linear(n_l1, 1))
  with torch.no_grad():
    np.random.seed(seed_num)
    net[0].weight.data = input_weight_init(n_inp, n_l1)
    net[1].weight = net[0].weight
    torch.nn.init.zeros_(net[2].weight)
    torch.nn.init.zeros_(net[2].bias)
  return net

def run_experiment(n_inp, n_tl1, T, n_l1, seed_num, target_seed):
  """Experiment with different number of features without search"""
  tnet = initialize_target_net(n_inp, n_tl1, target_seed)
  lossfunc = nn.MSELoss()
  net = initialize_learning_net(n_inp, n_l1, seed_num)
  sgd = optim.SGD(net[2:].parameters(), lr = 0.0)
  torch.manual_seed(seed_num + 2000)
  losses = []
  sample_average = 0.0
  with progressbar.ProgressBar(max_value=T) as bar:
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
      bar.update(t)
  losses = np.array(losses)
  return losses

def run_experiment_search(n_inp, n_tl1, T, n_l1, seed_num, target_seed):
  tnet = initialize_target_net(n_inp, n_tl1, target_seed, seed_num)
  lossfunc = nn.MSELoss()
  net = initialize_learning_net(n_inp, n_l1, seed_num)
  sgd = optim.SGD(net[2:].parameters(), lr = 0.0)
  torch.manual_seed(seed_num + 2000)
  losses = []
  ages = torch.zeros(n_l1)
  utils = torch.zeros(n_l1)
  sample_average = 0.0
  with progressbar.ProgressBar(max_value=T) as bar:
    for t in range(T):
      inp = torch.rand(n_inp)
      target = tnet(inp) + torch.randn(1)
      neck = net[:2](inp)
      pred = net[2:](neck)
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

      with torch.no_grad():
        ages += 1
        utils += 0.01*(torch.abs(net[2].weight.data[0]*neck) - utils)
        for i in range(n_l1//10):
          weak_node_i = torch.argmin(utils)
          weight_choice = [1.0,-1.0]
          net[0].weight[weak_node_i] = torch.from_numpy(np.random.choice(weight_choice, (net[0].weight.size()[1],)))
          net[0].bias[weak_node_i] = torch.randn(1)
          net[2].weight[0][weak_node_i] = 0.0
          utils[weak_node_i] = torch.median(utils)
          ages[weak_node_i] = 0

      bar.update(t)
  losses = np.array(losses)
  return losses


def main():
  parser = argparse.ArgumentParser(description="Test framework")
  parser.add_argument("-e", "--examples", type=int, default=30000,
                      help="no of examples to learn on")
  parser.add_argument("-n", "--runs", type=int, default=1,
                      help="number of runs")
  parser.add_argument("-i", "--input_size", type=int, default=10,
                      help="Input dimension")
  parser.add_argument("-f", "--features",  nargs='+', type=int, default=[100, 300, 1000],
                      help="Number of dimension(pass multiple)")
  parser.add_argument("-o", "--save", type=bool, default=False,
                      help="Saves the output graph")
  parser.add_argument("-s", "--seeds",  nargs='+', type=int, default=[1],
                      help="seeds in case of multiple runs")
  parser.add_argument("-t", "--target_seed", type=int, default=4000,
                      help="seed for choice of target net")
  args = parser.parse_args()
  T = args.examples
  n = args.runs 
  nbin = 200
  n_inp = args.input_size
  n_tl1 = 20
  n_feature = args.features
  n_seed = args.seeds
  t_seed = args.target_seed

  try:
    path = "output/out_" + str(t_seed)+".png"
    assert not os.path.exists(path)
  except:
    print("Experiment results already exist")
    sys.exit(1)

  try:
    assert t_seed not in n_seed
  except:
    print("Error: t_seed has to be different than n_seed")
    sys.exit(1)

  if (len(n_seed)!=n):
    print("Insuffcient number of seeds")
    return

  for nl_1 in n_feature:
    net_loss = 0
    print("No of Features:", nl_1)
    for l in range(n):
      print("Run:", l+1)
      net_loss = net_loss + run_experiment_search(n_inp, n_tl1, T, nl_1, n_seed[l], t_seed)
    net_loss = net_loss/n
    bin_losses = net_loss.reshape(T//nbin, nbin).mean(1)
    plt.plot(range(0, T, nbin), bin_losses, label=nl_1)
  tnet = initialize_target_net(n_inp, n_tl1, t_seed)
  norm_out = tnet[-1].weight.norm().data
  norm_out = format(float(norm_out), '.4f')
  title = "Output weight norm of t-net: " + str(norm_out) 
  plt.suptitle(title, fontsize=13)
  axes = plt.axes()
  axes.set_ylim([1.0, 3.5])
  plt.legend()

  if args.save:
    try:
      assert os.path.exists("output/")
    except:
      os.makedirs('output/') 
    filename = "output/out_" + str(t_seed)
    plt.savefig(filename)
  else:
    plt.show()


if __name__ == "__main__":
    main()
