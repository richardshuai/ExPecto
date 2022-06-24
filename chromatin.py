# -*- coding: utf-8 -*-
"""Compute chromatin representation of variants (required by predict.py).

This script takes a vcf file, compute the effect of the variant both at the
variant position and at nearby positions, and output the effects as the
representation that can be used by predict.py.

Example:
        $ python chromatin.py ./example/example.vcf

"""
import argparse
import math
import os

import h5py
import numpy as np
import pandas as pd
import pyfasta
import torch
from liftover import get_lifter
from torch import nn
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Predict variant chromatin effects')
parser.add_argument('inputfile', type=str, help='Input file in vcf format')
parser.add_argument('--hg38', action='store_true', help='Set flag if variants in VCF are given in hg38 format. '
                                                        'Variants will be lifted over to hg19.')
parser.add_argument("--chunk_size", action="store", dest="chunk_size", type=int, default=int(1e5), help="Size of chunks for batching predictions")
parser.add_argument("--chunk_i", action="store", dest="chunk_i", type=int, default=None, help="Chunk index for current run, starting from 0")
parser.add_argument('--maxshift', action="store",
                    dest="maxshift", type=int, default=800,
                    help='Maximum shift distance for computing nearby effects')
parser.add_argument('--inputsize', action="store", dest="inputsize", type=int,
                    default=2000, help="The input sequence window size for neural network")
parser.add_argument('--batchsize', action="store", dest="batchsize",
                    type=int, default=32, help="Batch size for neural network predictions.")
parser.add_argument('--output_dir', action="store", dest="output_dir",
                    type=str, default='chromatin_out', help="Output directory for predictions.")
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

genome = pyfasta.Fasta('./resources/hg19.fa')
os.makedirs(args.output_dir, exist_ok=True)

FAILED_LIFTOVER_VALUE = -1

if args.hg38:
    converter = get_lifter('hg38', 'hg19')  # need to liftover from hg38 to hg19

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

model = Beluga()
model.load_state_dict(torch.load('./resources/deepsea.beluga.pth'))
model.eval()
if args.cuda:
    model.cuda()

CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']


inputfile = args.inputfile
maxshift = args.maxshift
inputsize = args.inputsize
batchSize = args.batchsize
windowsize = inputsize + 100


def liftover_to_hg19(row):
    """
    Given pandas Series from vcf, return row with the hg38 pos lifted over to hg19. If we cannot liftover the variant
    successfully, return -1 for chrom and pos in the row.
    """
    chrom_hg38, pos_hg38 = row[0], row[1]
    hg19_coords = converter.convert_coordinate(chrom_hg38, pos_hg38)

    assert len(hg19_coords) <= 1, f"hg38 to hg19 conversion returned multiple entries for {chrom_hg38}, bp {pos_hg38}"

    if len(hg19_coords) == 0:
        chrom_hg19, pos_hg19 = FAILED_LIFTOVER_VALUE, FAILED_LIFTOVER_VALUE
    else:
        chrom_hg19, pos_hg19, _ = hg19_coords[0]
    row[0], row[1] = chrom_hg19, pos_hg19
    return row


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
    dataflip = seqsnp[:, ::-1, ::-1]
    seqsnp = np.concatenate([seqsnp, dataflip], axis=0)
    return seqsnp


def fetchSeqs(chr, pos, ref, alt, shift=0, inputsize=2000):
    """Fetches sequences from the genome.

    Retrieves sequences centered at the given position with the given inputsize.
    Returns both reference and alternative allele sequences . An additional 100bp
    is retrived to accommodate indels.

    Args:
        chr: the chromosome name that must matches one of the names in CHRS.
        pos: chromosome coordinate (1-based).
        ref: the reference allele.
        alt: the alternative allele.
        shift: retrived sequence center position - variant position.
        inputsize: the targeted sequence length (inputsize+100bp is retrived for
                reference allele).

    Returns:
        A string that contains sequence with the reference allele,
        A string that contains sequence with the alternative allele,
        A boolean variable that tells whether the reference allele matches the
        reference genome

        The third variable is returned for diagnostic purpose. Generally it is
        good practice to check whether the proportion of reference allele
        matches is as expected.

    """
    windowsize = inputsize + 100
    mutpos = int(windowsize / 2 - 1 - shift)
    # return string: ref sequence, string: alt sequence, Bool: whether ref allele matches with reference genome
    seq = genome.sequence({'chr': chr, 'start': pos + shift -
                           int(windowsize / 2 - 1), 'stop': pos + shift + int(windowsize / 2)})
    ref_matched_bool = seq[mutpos:(mutpos + len(ref))].upper() == ref.upper()
    alt_matched_bool = seq[mutpos:(mutpos + len(ref))].upper() == alt.upper()
    return seq[:mutpos] + ref + seq[(mutpos + len(ref)):], seq[:mutpos] + alt + seq[(mutpos + len(ref)):], ref_matched_bool, alt_matched_bool

vcf = pd.read_csv(inputfile, sep='\t', header=None, comment='#')

if args.chunk_i is not None:
    vcf = vcf.iloc[args.chunk_i * args.chunk_size:(args.chunk_i + 1) * args.chunk_size]

# lift over to hg19 if necessary
if args.hg38:
    print("Lifting over to hg38...")
    tqdm.pandas()
    vcf_lifted = vcf.progress_apply(liftover_to_hg19, axis=1)

    # Write variants not lifted over to VCF
    failed_liftover_mask = (vcf_lifted[1] == FAILED_LIFTOVER_VALUE)
    variants_not_lifted = vcf[failed_liftover_mask]
    print(f"Failed to lift {variants_not_lifted.shape[0]} variants from hg38 to hg19")
    variants_not_lifted.to_csv(f"{args.output_dir}/not_lifted.vcf", sep='\t', header=False, index=False)

    # Subset to lifted variants
    vcf = vcf_lifted[~failed_liftover_mask]

# Preserve vcf file with lifted coordinates
vcf_file_hg19 = f'{args.output_dir}/snps_hg19.vcf'
vcf_file_hg19_out = open(vcf_file_hg19, 'w')
print('##fileformat=VCFv4.3', file=vcf_file_hg19_out)
print('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO', file=vcf_file_hg19_out)
vcf_file_hg19_out.close()
vcf.to_csv(vcf_file_hg19, sep='\t', header=False, index=False, mode='a')

# standardize
vcf.iloc[:, 0] = 'chr' + vcf.iloc[:, 0].map(str).str.replace('chr', '')
vcf = vcf[vcf.iloc[:, 0].isin(CHRS)]

for shift in tqdm([0, ] + list(range(-200, -maxshift - 1, -200)) + list(range(200, maxshift + 1, 200))):
    refseqs = []
    altseqs = []
    ref_matched_bools = []
    alt_matched_bools = []
    for i in range(vcf.shape[0]):
        refseq, altseq, ref_matched_bool, alt_matched_bool = fetchSeqs(
            vcf.iloc[i, 0], int(vcf.iloc[i, 1]), vcf.iloc[i, 3], vcf.iloc[i, 4], shift=shift, inputsize=inputsize)
        refseqs.append(refseq)
        altseqs.append(altseq)
        ref_matched_bools.append(ref_matched_bool)
        alt_matched_bools.append(alt_matched_bool)

    if shift == 0:
        # only need to be checked once
        print(f"Number of variants with reference allele matched with reference genome: {np.sum(ref_matched_bools)}")
        print(f"Number of variants with alternate allele matched with reference genome: {np.sum(alt_matched_bools)}")
        print(f"Number of input variants: {len(ref_matched_bools)}")

    ref_encoded = encodeSeqs(refseqs, inputsize=inputsize).astype(np.float32)
    alt_encoded = encodeSeqs(altseqs, inputsize=inputsize).astype(np.float32)

    ref_preds = []
    for i in range(int(1 + (ref_encoded.shape[0]-1) / batchSize)):
        input = torch.from_numpy(ref_encoded[int(i*batchSize):int((i+1)*batchSize),:,:]).unsqueeze(2)
        if args.cuda:
            input = input.cuda()
        ref_preds.append(model.forward(input).cpu().detach().numpy().copy())
    ref_preds = np.vstack(ref_preds)

    alt_preds = []
    for i in range(int(1 + (alt_encoded.shape[0]-1) / batchSize)):
        input = torch.from_numpy(alt_encoded[int(i*batchSize):int((i+1)*batchSize),:,:]).unsqueeze(2)
        if args.cuda:
            input = input.cuda()
        alt_preds.append(model.forward(input).cpu().detach().numpy().copy())
    alt_preds = np.vstack(alt_preds)

    diff = alt_preds - ref_preds
    f = h5py.File(f'{args.output_dir}/snps.shift_{str(shift)}.diff.h5', 'w')
    f.create_dataset('diff', data=diff)
    f.create_dataset('ref', data=ref_preds)
    f.create_dataset('alt', data=alt_preds)
    f.close()
