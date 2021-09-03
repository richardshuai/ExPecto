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
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--targetIndex', action="store",
                    dest="targetIndex", type=int)
parser.add_argument('--output', action="store", dest="output")
parser.add_argument('--expFile', action="store", dest="expFile")
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
parser.add_argument('--plots_out_dir', type=str, default='plots')
parser.add_argument('--match_kidney_data_nans', action="store_true",
                    dest="match_kidney_data_nans", default=False,
                    help="If true, match nans of kidney data.")

args = parser.parse_args()

# read resources
Xreducedall = np.load(args.inputFile)
geneanno = pd.read_csv('./resources/geneanno.csv')

if args.filterStr == 'pc':
    filt = np.asarray(geneanno.iloc[:, -1] == 'protein_coding')
elif args.filterStr == 'lincRNA':
    filt = np.asarray(geneanno.iloc[:, -1] == 'lincRNA')
elif args.filterStr == 'all':
    filt = np.asarray(geneanno.iloc[:, -1] != 'rRNA')
else:
    raise ValueError('filterStr has to be one of all, pc, and lincRNA')

geneexp = pd.read_csv(args.expFile)
filt = filt * \
    np.isfinite(np.asarray(
        np.log(geneexp.iloc[:, args.targetIndex] + args.pseudocount)))

if args.match_kidney_data_nans:
    print("Using only genes found in kidney data...")
    kidney_exp_df = pd.read_csv('./resources/geneanno.exp_kidney.csv', index_col=0)
    filt = filt * ~np.array(np.any(kidney_exp_df.isnull(), axis=1))

# training

trainind = np.asarray(geneanno['seqnames'] != 'chrX') * np.asarray(
    geneanno['seqnames'] != 'chrY') * np.asarray(geneanno['seqnames'] != 'chr8')
testind = np.asarray(geneanno['seqnames'] == 'chr8')

dtrain = xgb.DMatrix(Xreducedall[trainind * filt, :])
dtest = xgb.DMatrix(Xreducedall[(testind) * filt, :])


dtrain.set_label(np.asarray(
    np.log(geneexp.iloc[trainind * filt, args.targetIndex] + args.pseudocount)))
dtest.set_label(np.asarray(
    np.log(geneexp.iloc[(testind) * filt, args.targetIndex] + args.pseudocount)))

param = {'booster': 'gblinear', 'base_score': args.base_score, 'alpha': 0,
         'lambda': args.l2, 'eta': args.eta, 'objective': 'reg:linear',
         'nthread': args.threads, "early_stopping_rounds": 10}

evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = args.num_round
bst = xgb.train(param, dtrain, num_round, evallist)
ypred = bst.predict(dtest)
ytrue = np.asarray(np.log(geneexp.iloc[(testind) * filt, args.targetIndex] + args.pseudocount))

print(spearmanr(ypred, ytrue))
if args.evalFile != '':
    evaldf = pd.DataFrame({'pred':ypred,'target':np.asarray(
     np.log(geneexp.iloc[(testind) * filt, args.targetIndex] + args.pseudocount))})
    evaldf.to_csv(args.evalFile)
bst.save_model(args.output + args.filterStr + '.pseudocount' + str(args.pseudocount) + '.lambda' + str(args.l2) + '.round' +
               str(args.num_round) + '.basescore' + str(args.base_score) + '.' + geneexp.columns[args.targetIndex] + '.save')
bst.dump_model(args.output + args.filterStr + '.pseudocount' + str(args.pseudocount) + '.lambda' + str(args.l2) + '.round' +
               str(args.num_round) + '.basescore' + str(args.base_score) + '.' + geneexp.columns[args.targetIndex] + '.dump')

# Plots
def plot_preds(ytrue, ypred, out_dir):
    print(spearmanr(ytrue, ypred))

    fig = sns.scatterplot(x=ytrue, y=ypred, color="black", alpha=0.3, s=20)
    plt.plot([0, 1], [0, 1], c='orange', transform=fig.transAxes)
    plt.xlim(np.min(ytrue), np.max(ytrue))
    plt.ylim(np.min(ytrue), np.max(ytrue))
    plt.ylabel('Predictions (log RPM)')
    plt.xlabel('Labels (log RPM)')
    train_pearsonr, _ = pearsonr(ytrue, ypred)
    train_r2 = r2_score(y_true=ytrue, y_pred=ypred)
    plt.title(f'PearsonR: {train_pearsonr:.3f}, R2: {train_r2:.3f}')
    plt.savefig(out_dir, dpi=300)

    plt.show()
    plt.close('all')


os.makedirs(args.plots_out_dir, exist_ok=True)
plot_preds(ytrue, ypred, f'{args.plots_out_dir}/test_plots.png')


ypred_train = bst.predict(dtrain)
ytrue_train = np.asarray(np.log(geneexp.iloc[trainind * filt, args.targetIndex] + args.pseudocount))
plot_preds(ytrue_train, ypred_train, f'{args.plots_out_dir}/train_plots.png')