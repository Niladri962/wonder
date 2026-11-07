"""Compare checkpoints on identical train/valid subsets: global WP and mean per-sequence WP.

Usage: python diag.py probe gru_full
"""
import sys

import torch
import numpy as np
import pyarrow.parquet as pq

from common import ART_DIR, TRAIN_PATH, VALID_PATH, WARMUP, Accumulator, Net, read_sequence


def load(name, device):
    ckpt = torch.load(ART_DIR / f"{name}.pt", map_location=device)
    cfg, sd = ckpt["cfg"], ckpt["state"]
    m = Net(sd["mean"], sd["std"], cfg["hidden"], cfg["layers"], cfg["d_in"],
            cfg.get("keep"), cfg.get("alphas", ())).to(device)
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def score(models, path, groups, masked, device):
    pf = pq.ParquetFile(str(path))
    accs = [Accumulator() for _ in models]
    per_seq = [[] for _ in models]
    for i in range(0, len(groups), 16):
        seqs = [read_sequence(pf, g, with_mask=masked) for g in groups[i:i + 16]]
        x = torch.from_numpy(np.stack([s[0] for s in seqs])).to(device)
        for k, m in enumerate(models):
            pred = m(x, x[:, 0])[0].cpu().numpy()
            for s, p in zip(seqs, pred):
                mask = s[2].copy() if masked else np.ones(len(p), bool)
                mask[:WARMUP] = False
                accs[k].add(s[1], p, mask)
                one = Accumulator()
                one.add(s[1], p, mask)
                per_seq[k].append(one.result()["wp"])
    return [(a.result()["wp"], float(np.mean(ps)), float(np.std([np.mean(p) for p in ps]) if False else 0))
            for a, ps in zip(accs, per_seq)]


def main():
    names = sys.argv[1:]
    device = torch.device("cuda")
    models = [load(n, device) for n in names]
    n_train = pq.ParquetFile(str(TRAIN_PATH)).num_row_groups
    train_groups = list(np.random.default_rng(1).permutation(n_train)[:200])  # first 200 seen by every run
    valid_groups = list(np.random.default_rng(5).permutation(1873)[:300])
    for label, path, groups, masked in [("train(200)", TRAIN_PATH, train_groups, False),
                                        ("valid(300)", VALID_PATH, valid_groups, True)]:
        for n, (g, ps, _) in zip(names, score(models, path, groups, masked, device)):
            print(f"{label:11s} {n:10s} global WP {g:.4f}   mean per-seq WP {ps:.4f}", flush=True)


if __name__ == "__main__":
    main()
