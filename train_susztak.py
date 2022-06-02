"""Training a ExPecto sequence-based expression model.

This script takes an expression profile, specified by the expression values
in the targetIndex-th column in expFile. The expression values can be
RPKM from RNA-seq. The rows
of the expFile must match with the genes or TSSes specified in
./resources/geneanno.csv.

Example:
        $ python ./train.py --expFile ./resources/geneanno.exp.csv --targetIndex 1 --output model.adipose


"""
import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from cluster_utils import get_keep_mask

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--expFile', action="store", dest="expFile")
parser.add_argument('--belugaFeatures', action="store", dest="belugaFeatures",
                    help="tsv file denoting Beluga features")
parser.add_argument('--inputFile', action="store",
                    dest="inputFile", default='./resources/Xreducedall.2002.npy')
parser.add_argument('--annoFile', action="store",
                    dest="annoFile", default='./resources/geneanno.csv')
parser.add_argument('--evalFile', action="store",
                     dest="evalFile", default='',help='specify to save holdout set predictions')
parser.add_argument('--filterStr', action="store",
                    dest="filterStr", type=str, default="all")
parser.add_argument('--pseudocount', action="store",
                    dest="pseudocount", type=float, default=0.0001)
parser.add_argument('--num_round', action="store",
                    dest="num_round", type=int, default=100)
parser.add_argument('--l2', action="store", dest="l2", type=float, default=100)
parser.add_argument('--l1', action="store", dest="l1", type=float, default=0)
parser.add_argument('--eta', action="store", dest="eta",
                    type=float, default=0.01)
parser.add_argument('--base_score', action="store",
                    dest="base_score", type=float, default=2)
parser.add_argument('--threads', action="store",
                    dest="threads", type=int, default=16)
parser.add_argument('--no_tf_features', action='store_true',
                    dest='no_tf_features', default=False,
                    help='leave out TF marks for training')
parser.add_argument('--no_dnase_features', action='store_true',
                    dest='no_dnase_features', default=False,
                    help='leave out DNase marks for training')
parser.add_argument('--no_histone_features', action='store_true',
                    dest='no_histone_features', default=False,
                    help='leave out histone marks for training')
parser.add_argument('--intersect_with_lambert', action='store_true',
                    dest='intersect_with_lambert', default=False,
                    help='intersect with Lambert2018_TFs_v_1.01_curatedTFs.csv')
parser.add_argument('--no_pol2', action='store_true',
                    dest='no_pol2', default=False,
                    help='take out Pol2*')
parser.add_argument('--output_dir', type=str, default='temp_expecto_model')

args = parser.parse_args()

# Make model output dir
os.makedirs(args.output_dir, exist_ok=True)
model_dir = f'{args.output_dir}/models'
os.makedirs(model_dir, exist_ok=True)

# read resources
Xreducedall = np.load(args.inputFile)
geneanno = pd.read_csv('./resources/geneanno.csv')

geneexp = pd.read_csv(args.expFile)

pearsonr_valids = []
r2_valids = []
pearsonr_trains = []
r2_trains = []

for ti in range(1, len(geneexp.columns)):
    print(f"Cell type: {geneexp.columns[ti]}")

    if args.filterStr == 'pc':
        filt = np.asarray(geneanno.iloc[:, -1] == 'protein_coding')
    elif args.filterStr == 'lincRNA':
        filt = np.asarray(geneanno.iloc[:, -1] == 'lincRNA')
    elif args.filterStr == 'all':
        filt = np.asarray(geneanno.iloc[:, -1] != 'rRNA')
    else:
        raise ValueError('filterStr has to be one of all, pc, and lincRNA')

    filt = filt * \
        np.isfinite(np.asarray(
            np.log(geneexp.iloc[:, ti] + args.pseudocount)))

    # Ablations
    beluga_features_df = pd.read_csv(args.belugaFeatures, sep='\t', index_col=0)
    beluga_features_df['Assay type + assay + cell type'] = beluga_features_df['Assay type'] + '/' + beluga_features_df[
        'Assay'] + '/' + beluga_features_df['Cell type']

    keep_mask = get_keep_mask(beluga_features_df, args.no_tf_features, args.no_dnase_features,
                  args.no_histone_features, args.intersect_with_lambert, args.no_pol2)
    keep_indices = np.nonzero(keep_mask)[0]
    num_genes = Xreducedall.shape[0]
    Xreducedall = Xreducedall.reshape(num_genes, 10, 2002)[:, :, keep_indices].reshape(num_genes, -1)

    print(f'Training data shape: {Xreducedall.shape}')

    # training
    train_ind = np.asarray(geneanno['seqnames'] != 'chrX') * \
                np.asarray(geneanno['seqnames'] != 'chrY') * \
                np.asarray(geneanno['seqnames'] != 'chr7') * \
                np.asarray(geneanno['seqnames'] != 'chr8')

    val_ind = np.asarray(geneanno['seqnames'] == 'chr8')

    dtrain = xgb.DMatrix(Xreducedall[train_ind * filt, :])
    dval = xgb.DMatrix(Xreducedall[val_ind * filt, :])

    dtrain.set_label(np.asarray(
        np.log(geneexp.iloc[train_ind * filt, ti] + args.pseudocount)))
    dval.set_label(np.asarray(
        np.log(geneexp.iloc[val_ind * filt, ti] + args.pseudocount)))

    param = {'booster': 'gblinear', 'base_score': args.base_score, 'alpha': 0,
             'lambda': args.l2, 'eta': args.eta, 'objective': 'reg:linear',
             'nthread': args.threads, "early_stopping_rounds": 10}

    evallist = [(dval, 'eval'), (dtrain, 'train')]
    num_round = args.num_round
    bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)

    bst.save_model(f'{model_dir}/expecto_{args.filterStr}.pseudocount{args.pseudocount}.lambda{args.l2}'
                   f'.round{args.num_round}.basescore{args.base_score}.{geneexp.columns[ti]}.save')
    bst.dump_model(f'{model_dir}/expecto_{args.filterStr}.pseudocount{args.pseudocount}.lambda{args.l2}'
                   f'.round{args.num_round}.basescore{args.base_score}.{geneexp.columns[ti]}.dump')

    # Plots
    def plot_preds(ytrue, ypred, out_dir):
        fig = sns.scatterplot(x=ytrue, y=ypred, color="black", alpha=0.3, s=20)
        plt.plot([0, 1], [0, 1], c='orange', transform=fig.transAxes)
        plt.xlim(np.min(ytrue), np.max(ytrue))
        plt.ylim(np.min(ytrue), np.max(ytrue))
        plt.ylabel('Predictions (log RPM)')
        plt.xlabel('Labels (log RPM)')
        pearsonr_value, _ = pearsonr(ytrue, ypred)
        r2_value = r2_score(y_true=ytrue, y_pred=ypred)
        plt.title(f'PearsonR: {pearsonr_value:.3f}, R2: {r2_value:.3f}')
        plt.savefig(out_dir, dpi=300)

        plt.show()
        plt.close('all')
        return pearsonr_value, r2_value

    ypred_val = bst.predict(dval)
    ytrue_val = np.asarray(np.log(geneexp.iloc[val_ind * filt, ti] + args.pseudocount))
    pearsonr_val, r2_val = plot_preds(ytrue_val, ypred_val, f'{args.output_dir}/{ti}_val_plot.png')
    pearsonr_valids.append(pearsonr_val)
    r2_valids.append(r2_val)

    ypred_train = bst.predict(dtrain)
    ytrue_train = np.asarray(np.log(geneexp.iloc[train_ind * filt, ti] + args.pseudocount))
    pearsonr_train, r2_train = plot_preds(ytrue_train, ypred_train, f'{args.output_dir}/{ti}_train_plots.png')
    pearsonr_trains.append(pearsonr_train)
    r2_trains.append(r2_train)

metrics_dir = f'{args.output_dir}/metrics'
os.makedirs(metrics_dir, exist_ok=True)

with h5py.File(f'{metrics_dir}/metrics.h5', 'w') as h5_out:
    h5_out.create_dataset('pearsonr_valids', data=pearsonr_valids)
    h5_out.create_dataset('r2_valids', data=r2_valids)
    h5_out.create_dataset('pearsonr_trains', data=pearsonr_trains)
    h5_out.create_dataset('r2_trains', data=r2_trains)
