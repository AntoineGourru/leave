import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

filepath = sys.argv[1]
loss = sys.argv[2]
get_plot = bool(sys.argv[3])
saveplot = sys.argv[4]


def parse_line(row):
    """
    Parser to read the result files
    :param row: the row of the file
    :return: the parsed row
    """
    row = row.replace("\t", ",")
    m = {}
    for part in row.split(","):
        name, value = part.split("=")
        if name == "TIME":
            continue
        m[name] = value
    return m


# Versions with VIB
def extract_best(res):

    f = pd.DataFrame(res)
    if loss == 'woVIB':
        f.columns = ['ALPHA', 'LR', 'EPOCH', 'ENCODER', 'HIDDEN CHANNELS', 'NUM LAYERS', 'SEED', 'AUC', 'DI', 'EO',
                     'RB', 'DURATION']
    else:
        f.columns = ['ALPHA', 'LR', 'EPOCH', 'ENCODER', 'HIDDEN CHANNELS', 'NUM LAYERS', 'L', 'BETA', 'SEED',
                     'AUC', 'DI', 'EO', 'RB', 'DURATION']

    f.reset_index(inplace=True)
    f.drop("index", axis=1, inplace=True)

    f = f.astype({'ALPHA': 'float', 'AUC': 'float', 'NUM LAYERS': 'int', 'DI': 'float', 'EO': 'float', 'RB': 'float',
                  'LR': 'float', 'EPOCH': 'int'})

    if loss == 'VIB':
        f = f.astype({'BETA': 'float', 'L': 'int'})

    # Compute 1/DI (IDI), the mean between DI and AUC, same then harmonic mean
    f['1-RB'] = 1 - f['RB']
    f['IDI'] = 1/f['DI']
    f['avg_RB_AUC'] = f[['AUC', '1-RB']].mean(axis=1)
    f['avg_DI_AUC'] = f[['AUC', 'DI']].mean(axis=1)
    f['fscore'] = 2 * (f['AUC'] * f['DI']) / (f['AUC'] + f['DI'])

    res_filtered = f

    if loss == 'woVIB':
        # Average on seeds
        mean = res_filtered.groupby(['ALPHA', 'HIDDEN CHANNELS', 'EPOCH', 'LR', 'NUM LAYERS'],
                                as_index=False)[['avg_DI_AUC', 'fscore', 'AUC', 'DI', 'IDI', 'EO', 'RB']].mean()

        # Standard deviation on seeds
        std = res_filtered.groupby(['ALPHA', 'HIDDEN CHANNELS', 'EPOCH', 'LR', 'NUM LAYERS'],
                                   as_index=False)[['avg_DI_AUC', 'fscore', 'AUC', 'DI', 'IDI', 'EO', 'RB']].std()
    else:
        mean = res_filtered.groupby(['ALPHA', 'HIDDEN CHANNELS', 'EPOCH', 'LR', 'NUM LAYERS', 'L', 'BETA'],
                                    as_index=False)[['avg_DI_AUC', 'fscore', 'AUC', 'DI', 'IDI', 'EO', 'RB']].mean()

        std = res_filtered.groupby(['ALPHA', 'HIDDEN CHANNELS', 'EPOCH', 'LR', 'BETA', 'L'],
                                   as_index=False)[['avg_DI_AUC', 'fscore', 'AUC', 'DI', 'IDI', 'EO', 'RB']].std()

    # Get the best set of hyper-parameter
    best_set = mean[mean.avg_DI_AUC == mean.avg_DI_AUC.max()]
    print(best_set)

    # Extract all scores for this particular set of hyper-parameters (woVIB)
    _temp1 = mean[(mean['LR'] == best_set.iloc[0]['LR']) &
                  (mean['HIDDEN CHANNELS'] == best_set.iloc[0]['HIDDEN CHANNELS']) & (
                          mean['NUM LAYERS'] == best_set.iloc[0]['NUM LAYERS'])]
    _temp2 = std[(std['LR'] == best_set.iloc[0]['LR']) &
                 (std['HIDDEN CHANNELS'] == best_set.iloc[0]['HIDDEN CHANNELS']) & (
                         std['NUM LAYERS'] == best_set.iloc[0]['NUM LAYERS'])]

    return _temp1, _temp2, best_set


results = []

with open(filepath, encoding="latin-1") as f:
    for line in f:
        info = parse_line(line)
        results.append(info)

avg, std, best_config = extract_best(results)

if get_plot:
    # Visualisation of the impact of alpha
    Nsteps, Nwalkers = 10, 10
    t = np.arange(Nsteps)

    # Nsteps length arrays empirical means and standard deviations of both
    # populations over time
    logDI = np.log(1/np.array(avg.DI))
    DIsigma = np.std(np.array(logDI))

    AUC = np.array(avg.AUC)
    AUCsigma = np.array(std.AUC)

    RB = np.array(avg.RB)
    RBsigma = np.array(std.RB)

    label_size = 12
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    # plot it!
    fig, ax = plt.subplots(1)
    ax.plot(t, logDI, lw=2, label='IDI', color='dodgerblue')
    ax.plot(t, AUC, lw=2, label='AUC', color='limegreen')
    ax.plot(t, RB, lw=2, label='RB', color='orange')
    ax.fill_between(t, logDI + DIsigma, logDI - DIsigma, facecolor='dodgerblue', alpha=0.5)
    ax.fill_between(t, AUC + AUCsigma, AUC - AUCsigma, facecolor='limegreen', alpha=0.5)
    ax.fill_between(t, RB + RBsigma, RB - RBsigma, facecolor='orange', alpha=0.5)
    ax.legend(loc='lower left', fontsize=16)
    ax.set_xlabel(r'$\alpha$', fontsize=18)
    ax.set_ylabel('Score', fontsize=18)
    ax.grid()
    plt.savefig(saveplot+'.pdf', dpi=300, bbox_inches='tight')
    plt.show()

