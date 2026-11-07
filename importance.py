"""Permutation importance of feature groups on validation half B (CPU only).

Each group is replaced by the same columns from a different sequence, which
keeps realistic values and dynamics but breaks their link to the targets.
Usage: python importance.py gru_mask 60
"""
import sys
import time

import torch
import numpy as np
import pyarrow.parquet as pq

from common import FEATURE_COLUMNS, VALID_PATH, WARMUP, Accumulator, read_sequence
from diag import load
from mask_model import valid_halves

GROUPS = {}
for inst in range(2):
    base = 52 * inst
    GROUPS[f"i{inst} bid price p0-10"] = range(base, base + 11)
    GROUPS[f"i{inst} ask price p11-21"] = range(base + 11, base + 22)
    GROUPS[f"i{inst} bid volume v0-10"] = range(base + 22, base + 33)
    GROUPS[f"i{inst} ask volume v11-21"] = range(base + 33, base + 44)
    GROUPS[f"i{inst} trade price dp"] = range(base + 44, base + 48)
    GROUPS[f"i{inst} trade volume dv"] = range(base + 48, base + 52)
GROUPS["i1 (whole instrument)"] = range(52, 104)
for k in range(8):
    GROUPS[f"a{k}"] = [104 + k]


@torch.no_grad()
def score(model, xs, ys, ms):
    acc = Accumulator()
    for i in range(0, len(xs), 6):
        x = torch.from_numpy(xs[i:i + 6])
        pred = model(x, x[:, 0])[0].numpy()
        for p, y, m in zip(pred, ys[i:i + 6], ms[i:i + 6]):
            acc.add(y, p, m)
    return acc.result()["wp"]


def main():
    name, n = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(4)  # leave cores for the training data loader
    model = load(name, torch.device("cpu"))
    pf = pq.ParquetFile(str(VALID_PATH))
    groups = valid_halves()[1][:n]
    seqs = [read_sequence(pf, g, with_mask=True) for g in groups]
    xs = np.stack([s[0] for s in seqs])
    ys = np.stack([s[1] for s in seqs])
    ms = np.stack([s[2] for s in seqs])
    ms[:, :WARMUP] = False
    t = time.time()
    base = score(model, xs, ys, ms)
    print(f"base WP on {n} half-B sequences: {base:.4f} ({time.time() - t:.0f}s per pass)", flush=True)
    donor = np.roll(np.arange(n), 1)  # each sequence borrows the group from its neighbour
    rows = []
    for gname, cols in GROUPS.items():
        cols = list(cols)
        xp = xs.copy()
        xp[:, :, cols] = xs[donor][:, :, cols]
        drop = base - score(model, xp, ys, ms)
        rows.append((drop, gname))
        print(f"  {gname:24s} ({len(cols):2d} cols)  WP drop {drop:+.4f}", flush=True)
    print("\nranked (biggest drop = most useful):")
    for drop, gname in sorted(rows, reverse=True):
        print(f"  {drop:+.4f}  {gname}")


if __name__ == "__main__":
    main()
