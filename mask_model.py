"""Learn P(is_scored | features so far) on validation half A; check AUC on half B.

The frozen model later reweights training rows toward moments that look like
scored ones. It never sees targets.
Usage: python mask_model.py
"""
import time

import torch
import torch.nn as nn
import numpy as np
import pyarrow.parquet as pq

from common import ART_DIR, N_FEATURES, VALID_PATH, WARMUP, read_sequence

N_VALID = 1873


def valid_halves():
    perm = np.random.default_rng(7).permutation(N_VALID)
    return sorted(perm[:N_VALID // 2].tolist()), sorted(perm[N_VALID // 2:].tolist())


class MaskNet(nn.Module):
    def __init__(self, mean, std, hidden=64):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.gru = nn.GRU(N_FEATURES, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        xn = torch.clamp((x - self.mean) / self.std, -8, 8)
        return self.head(self.gru(xn)[0]).squeeze(-1)  # logits [B,T]


def load_all(groups):
    pf = pq.ParquetFile(str(VALID_PATH))
    xs, ms = [], []
    for g in groups:  # float16 keeps ~900 sequences within RAM
        x, _, m = read_sequence(pf, g, with_mask=True)
        xs.append(x.astype(np.float16))
        ms.append(m)
    return np.stack(xs), np.stack(ms)


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.sum()
    return (ranks[label].sum() - pos * (pos + 1) / 2) / (pos * (len(label) - pos))


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    half_a, half_b = valid_halves()
    t = time.time()
    xa, ma = load_all(half_a)
    xb, mb = load_all(half_b[:300])
    print(f"loaded in {time.time() - t:.0f}s", flush=True)
    flat = xa[:, ::10].astype(np.float32).reshape(-1, N_FEATURES).astype(np.float64)
    std = flat.std(0)
    std[std < 1e-6] = 1
    model = MaskNet(flat.mean(0), std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    pos_weight = torch.tensor((1 - ma.mean()) / ma.mean(), device=device)
    rng = np.random.default_rng(0)
    for epoch in range(6):
        losses = []
        for idx in np.array_split(rng.permutation(len(xa)), len(xa) // 32):
            x = torch.from_numpy(xa[idx]).to(device).float()
            m = torch.from_numpy(ma[idx]).float().to(device)
            logit = model(x)[:, WARMUP:]
            loss = nn.functional.binary_cross_entropy_with_logits(logit, m[:, WARMUP:], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            pb = torch.cat([torch.sigmoid(model(torch.from_numpy(xb[i:i + 32]).to(device).float())).cpu()
                            for i in range(0, len(xb), 32)]).numpy()[:, WARMUP:].ravel()
        lb = mb[:, WARMUP:].ravel()
        print(f"epoch {epoch} loss {np.mean(losses):.4f} half-B AUC {auc(pb, lb):.4f} "
              f"mean p scored {pb[lb].mean():.3f} unscored {pb[~lb].mean():.3f}", flush=True)
    torch.save(model.state_dict(), ART_DIR / "maskgru.pt")


if __name__ == "__main__":
    main()
