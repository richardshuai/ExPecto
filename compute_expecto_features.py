# -*- coding: utf-8 -*-
import argparse
import math
import pyfasta
import torch
from torch import nn
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import os
from liftover import get_lifter

# Script based on https://github.com/FunctionLab/ExPecto/issues/9


def main():
    parser = argparse.ArgumentParser(description='Compute ExPecto chromatin features for TSS list')
    parser.add_argument('annoFile')
    parser.add_argument('tss_file')
    parser.add_argument('--windowsize', action="store",
                        dest="windowsize", type=int, default=2000,
                        help='Input window size for predictions')
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_compute_expecto_features',
                        help='Output directory')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()


    os.makedirs(args.out_dir, exist_ok=True)
    genome = pyfasta.Fasta('./resources/hg19.fa')

    # start by reading in the .npy features
    npy_features_file = "./resources/Xreducedall.2002.npy"

    model = Beluga()
    model.load_state_dict(torch.load('./resources/deepsea.beluga.pth'))
    model.eval()
    if args.cuda:
        model.cuda()

    # Read in gene anno file and get TSSs
    tss_df = pd.read_csv(args.tss_file, sep='\t', index_col=0)
    tss_df = tss_df.set_index('ens_id')
    converter = get_lifter('hg38', 'hg19')  # need to liftover from hg38 to hg19

    gene_chrom_tss_strand = []

    genes_found = 0
    no_mappings = 0
    for i, line in enumerate(open(args.annoFile)):
        if i == 0:
            continue
        gene_id, _, chrom, strand, _, representative_tss, _ = line.rstrip().split(",")
        if gene_id in tss_df.index:
            genes_found += 1
            chrom_hg38, representative_tss_hg38, strand, _, is_default_tss = tss_df.loc[gene_id]
            hg19_coords = converter.convert_coordinate(chrom_hg38, representative_tss_hg38)

            if len(hg19_coords) == 0:
                # if no mapping returned for some reason, just use original as annotated in ExPecto
                no_mappings += 1
            elif not is_default_tss:
                # otherwise, only use new TSS if atac counts are not all zero around all TSSs
                assert len(hg19_coords) == 1, f"hg38 to hg19 conversion returned multiple entries for {chrom_hg38}," \
                                          f"position {representative_tss_hg38}"
                chrom, representative_tss, _ = converter.convert_coordinate(chrom_hg38, representative_tss_hg38)[0]

        gene_chrom_tss_strand.append((gene_id, chrom, int(representative_tss), (1 if strand == "+" else -1)))

    print(f"Found {genes_found} genes in geneAnno file that match a TSS in provided TSS file...")
    print(f"Failed to convert {no_mappings} hg38 positions to hg19 with liftover tool...")
    # Read in annoFile as pd df
    geneanno_df = pd.read_csv(args.annoFile, index_col=0)
    changed_tss_count = 0
    for gene, chrom, tss, strand in gene_chrom_tss_strand:

        if geneanno_df.loc[gene, 'CAGE_representative_TSS'] != tss:
            # print(tss_df.loc[gene])
            # print((gene, chrom, tss))
            # print(geneanno_df.loc[gene])
            # print('===================================')
            changed_tss_count += 1
    print(f"Found {changed_tss_count} altered TSSs out of {geneanno_df.shape[0]} total TSSs...")

    # Weights for computing features
    windowsize = args.windowsize
    shifts = np.array(list(range(-20000, 20000, 200)))

    pos_weight_shifts = shifts
    pos_weights = np.vstack([
        np.exp(-0.01 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
        np.exp(-0.02 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
        np.exp(-0.05 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
        np.exp(-0.1 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
        np.exp(-0.2 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
        np.exp(-0.01 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
        np.exp(-0.02 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
        np.exp(-0.05 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
        np.exp(-0.1 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
        np.exp(-0.2 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0)])

    # Make predictions and compute features with weights
    computed_features_with_rc = []
    for gene, chrom, tss, strand in tqdm(gene_chrom_tss_strand):
        seqs_to_predict = []
        for shift in shifts:
            seq = genome.sequence({'chr': chrom,
                                   'start': tss + (shift * strand) - int(windowsize / 2 - 1),
                                   'stop': tss + (shift * strand) + int(windowsize / 2)})
            seqs_to_predict.append(seq)

        seqsnp = encodeSeqs(seqs_to_predict)

        model_input = torch.from_numpy(np.array(seqsnp)).unsqueeze(2).float()
        rc_model_input = torch.from_numpy(np.array(seqsnp[:, ::-1, ::-1])).unsqueeze(2).float()

        if args.cuda:
            model_input = model_input.cuda()
            rc_model_input = rc_model_input.cuda()
        prediction = model.forward(model_input).detach().cpu().numpy().copy()
        rc_prediction = model.forward(rc_model_input).detach().cpu().numpy().copy()
        pred_fwd_rc = 0.5 * (prediction + rc_prediction)
        computed_features_with_rc.append(np.sum(pos_weights[:, :, None] * pred_fwd_rc[None, :, :], axis=1).flatten())

    computed_features_with_rc = np.array(computed_features_with_rc)

    np.save(f'{args.out_dir}/Xreducedall.2002.representative_tss_top', computed_features_with_rc)


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def encodeSeqs(seqs, inputsize=2000):
    """Convert sequences to 0-1 encoding and truncate to the input size.
    The output concatenates the forward and reverse complement sequence
    encodings.

    Args:
        seqs: list of sequences (e.g. produced by fetchSeqs)
        inputsize: the number of basepairs to encode in the output

    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize

    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """
    seqsnp = np.zeros((len(seqs), 4, inputsize), np.bool_)

    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}

    n = 0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = mydict[c]
        n = n + 1

    # get the complementary sequences
    # dataflip = seqsnp[:, ::-1, ::-1]
    # seqsnp = np.concatenate([seqsnp, dataflip], axis=0)
    return seqsnp


if __name__ == '__main__':
    main()
