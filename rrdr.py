import numpy as np
import pickle
import math
import statistics
import matplotlib.pyplot as plt
import argparse

def read_losses(features, seed_num, search=False, pathstr=''):
    if search:
        path = pathstr + 'search/' + str(features) + '/'  
    else:
        path = pathstr + 'fixed/' + str(features) + '/'
    fname = path + 'run_' + str(seed_num)
    dbfile = open(fname, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db

def calculate_average(features, seeds,  search=False, pathstr=''):
    net_loss=[]
    for seed_num in seeds:
        if search:
            net_loss1 = read_losses(features, seed_num, search=True, pathstr=pathstr) 
            net_loss.append(statistics.mean(net_loss1[-10000:]))
        else:
            net_loss1 = read_losses(features, seed_num, pathstr=pathstr)
            net_loss.append(statistics.mean(net_loss1[-10000:]))
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
    nbin = 10000
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
            fig1, ax1 = plt.subplots()
            i=0
            rrs = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
            rrs1 = rrs
            for dr in ['0.1','0.01', '0.001', '0.0001']:
                stds =[]
                means =[]
                for rr in ['0.010000', '0.001000', '0.000100', '0.000010', '0.000001', '0.000000']:
                    pathstr = 'experiment_relu/ReLUpaper_rr' + rr +'_dr'+ dr +'/'
                    net_loss = calculate_average(features, n_seed, search=True, pathstr=pathstr)
                    m = len(net_loss)
                    netl = np.array(net_loss)
                    mean_1 = netl.mean()
                    means.append(mean_1)
                    stds.append(netl.std()/math.sqrt(m))
                    
                label = dr
                if(dr=='f'):
                    label='fixed'
                if(dr=='10'):
                    label='1'
                rrs =np.array(rrs)
                means = np.array(means)
                stds = np.array(stds)
                ax1.plot(rrs, means, label=label)
                plt.fill_between(rrs, means-stds, means+stds, alpha=0.4)
                i=i+1
    
    ax1.set_xscale('log')
    ax1.set_xticks(rrs1)
    lable = ['10^-2', '10^-3', '10^-4', '10^-5', '10^-6',  'fixed']
    ax1.set_xticklabels(lable)
    ax1.get_xaxis().get_major_formatter().labelOnlyBase = False
    plt.xlabel('Replacement Rate')
    plt.ylabel('Mean Squared Error(Last 10k samples)')
    plt.legend()
    plt.grid()
    plt.savefig("RelU100rrdrfinal.svg", format="svg")

if __name__ == "__main__":
    main()
 
