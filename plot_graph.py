import numpy as np
import pickle
import statistics
import matplotlib.pyplot as plt
import argparse
import math
def read_losses(features, seed_num, search=False):
    if search:
        path = 'LTU_rr0.005000_dr0.01/search/' + str(features) + '/'  
    else:
        path = 'LTU_rr0.005000_dr0.01/fixed/' + str(features) + '/'
    fname = path + 'run_' + str(seed_num)
    dbfile = open(fname, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db

def calculate_average(features, seeds,  search=False):
    net_loss = 0
    for seed_num in seeds:
        if search:
            net_loss = net_loss + read_losses(features, seed_num, search=True)
        else:
            net_loss = net_loss + read_losses(features, seed_num)
    net_loss = net_loss/len(seeds)
    return net_loss

def main():
    parser = argparse.ArgumentParser(description="Plots Graphs from losses")
    parser.add_argument("-se", "--search", action='store_true',
                          help="run experiment with search")
    parser.add_argument("-f", "--features",  nargs='+', type=int, default=[100, 300, 1000],
                        help="Number of features")
    parser.add_argument("-e", "--examples", type=int, default=30000,
                      help="no of examples used in experiment")
    parser.add_argument("-s", "--seeds",  nargs='+', type=int, default=[1],
                        help="seeds used in experiment")            
    parser.add_argument("-p", "--plot_all", action='store_true',
                          help="Plots all graphs")
    nbin = 1000
    args = parser.parse_args()
    T = args.examples
    n_feature = args.features
    n_seed = args.seeds
    plt_all = args.plot_all

    for features in n_feature:
        if args.search and not plt_all:
            net_loss = calculate_average(features, n_seed, search=True)
            bin_losses = net_loss.reshape(T//nbin, nbin).mean(1)
            label = str(features) + "-s"
            plt.plot(range(0, T, nbin), bin_losses, label=label)
        elif not plt_all:
            net_loss = calculate_average(features, n_seed)
            bin_losses = net_loss.reshape(T//nbin, nbin).mean(1)
            label = str(features) + "-f"
            plt.plot(range(0, T, nbin), bin_losses, label=label)
        else:
            net_loss = calculate_average(features, n_seed, search=True)
            stds = net_loss.std()/math.sqrt(30)
            bin_losses = net_loss.reshape(T//nbin, nbin).mean(1)
            label = str(features) + "-s"
            plt.plot(range(0, T, nbin), bin_losses, label=label)
            plt.fill_between(range(0, T, nbin), bin_losses-stds, bin_losses+stds, alpha=0.4)
            net_loss = calculate_average(features, n_seed)
            stdf = net_loss.std()/math.sqrt(30)
            bin_losses = net_loss.reshape(T//nbin, nbin).mean(1)
            label = str(features) + "-f"
            plt.plot(range(0, T, nbin), bin_losses, label=label)
            plt.fill_between(range(0, T, nbin), bin_losses-stds, bin_losses+stds, alpha=0.4)

    axes = plt.axes()
    axes.set_ylim([1.0, 3.5])
    plt.legend()
    plt.xlabel('Number of Samples')
    plt.ylabel('Mean Squared Error(MSE)')
    plt.savefig("paper1.svg", format="svg")

if __name__ == "__main__":
    main()
