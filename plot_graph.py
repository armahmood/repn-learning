import numpy as np
import pickle
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt


def read_losses(features, seed_num, search=False):
    if search:
        path = 'losses/search/' + str(features) + '/'  
    else:
        path = 'losses/fixed/' + str(features) + '/'
    fname = path + 'run_' + str(seed_num)
    dbfile = open(fname, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db

def calculate_average(n_runs, features, seed_num,  search=False):
    net_loss = 0
    for _ in range(n_runs):
        if search:
            net_loss = net_loss + read_losses(features, seed_num, search=True)
        else:
            net_loss = net_loss + read_losses(features, seed_num)
    net_loss = net_loss/n_runs
    return net_loss

def plot_meanerror(n_runs, seed_num, feature_list, T, search=False):
    nbin = 200
    for feature in feature_list: 
        if search:
            net_loss = calculate_average(n_runs, seed_num, feature, search=True)
        else:
            net_loss = calculate_average(n_runs, seed_num, feature)
        bin_losses = net_loss.reshape(T//nbin, nbin).mean(1)
        plt.plot(range(0, T, nbin), bin_losses, label=feature)
    axes = plt.axes()
    axes.set_ylim([1.0, 3.5])
    plt.legend()
