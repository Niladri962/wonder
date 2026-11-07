"""Shared data loading, model definition and metric helpers for training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import threading

# torch must load its DLLs before pyarrow on Windows (WinError 1114 otherwise).
import torch
import torch.nn as nn
import numpy as np
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get(
    "WNN_DATA", r"C:\Users\Niladri Ghosh\wunder_data\wnn_connectome_starterpack"))
TRAIN_PATH = DATA_ROOT / "datasets" / "train.parquet"
VALID_PATH = DATA_ROOT / "datasets" / "valid.parquet"
ART_DIR = Path(__file__).resolve().parent / "artifacts"

SEQ_LEN = 20_000
WARMUP = 99
N_FEATURES = 112
FEATURE_COLUMNS = [f"i{i}_{g}{j}" for i in range(2)
                   for g, n in [("p", 22), ("v", 22), ("dp", 4), ("dv", 4)] for j in range(n)]
FEATURE_COLUMNS += [f"a{i}" for i in range(8)]
TARGET_COLUMNS = ["t0", "t1"]
CLIP = 2.0


def read_sequence(pf: pq.ParquetFile, group: int, with_mask: bool = False):
    """Return (features[20000,112] float32, targets[20000,2] float32[, is_scored])."""
    cols = FEATURE_COLUMNS + TARGET_COLUMNS + (["is_scored"] if with_mask else [])
    table = pf.read_row_group(group, columns=cols, use_threads=False)
    x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
    if with_mask:
        return x, y, table["is_scored"].to_numpy().astype(bool)
    return x, y


class ThreadedBatches:
    """Yield (x[B,T,112], y[B,T,2]) batches of whole sequences, read by a thread pool.

    Threads instead of DataLoader worker processes: on Windows every worker
    re-imports torch/CUDA and the commit charge exhausts the paging file.
    pyarrow releases the GIL while decoding, so threads still read in parallel.
    """

    def __init__(self, items, batch, threads=6, prefetch=2, seed=0):
        # items: list of (parquet path, row group) so several files can be mixed.
        self.items, self.batch = [(str(p), int(g)) for p, g in items], batch
        self.threads, self.prefetch = threads, prefetch
        self.rng = np.random.default_rng(seed)
        self.local = threading.local()

    def __len__(self):
        return len(self.items) // self.batch

    def _read(self, item):
        path, g = item
        files = self.local.__dict__.setdefault("files", {})
        if path not in files:
            files[path] = pq.ParquetFile(path)
        return read_sequence(files[path], g)

    def _load(self, pool, groups):
        seqs = list(pool.map(self._read, groups))
        return (torch.from_numpy(np.stack([s[0] for s in seqs])).pin_memory(),
                torch.from_numpy(np.stack([s[1] for s in seqs])).pin_memory())

    def __iter__(self):
        order = self.rng.permutation(len(self.items))
        batches = [[self.items[j] for j in order[i * self.batch:(i + 1) * self.batch]]
                   for i in range(len(self))]
        with ThreadPoolExecutor(self.threads) as pool, ThreadPoolExecutor(self.prefetch) as outer:
            pending = [outer.submit(self._load, pool, b) for b in batches[:self.prefetch]]
            for i in range(len(batches)):
                result = pending.pop(0).result()
                if i + self.prefetch < len(batches):
                    pending.append(outer.submit(self._load, pool, batches[i + self.prefetch]))
                yield result


def ema_scan(xn, alpha, L=250):
    """Exact causal EMA over the time axis of xn [B,T,F], started at xn[:,0].

    Chunked: a dense L x L decay matrix inside each chunk, a short loop to carry
    the state between chunks. Matches  e = x0;  e += alpha * (x_t - e)  step by step.
    """
    B, T, F = xn.shape
    i = torch.arange(L, device=xn.device, dtype=torch.float32)
    lag = i[:, None] - i[None, :]
    W = torch.where(lag >= 0, alpha * (1 - alpha) ** lag.clamp(min=0), torch.zeros_like(lag))
    d = (1 - alpha) ** (i + 1)
    inner = torch.einsum("ij,bcjf->bcif", W, xn.reshape(B, T // L, L, F))
    out, prev = [], xn[:, 0]
    for c in range(T // L):
        e = inner[:, c] + d[None, :, None] * prev[:, None, :]
        out.append(e)
        prev = e[:, -1]
    return torch.cat(out, dim=1)


class Net(nn.Module):
    """Normalise -> [select columns, add EMAs] -> input MLP -> stacked GRU -> linear head.

    Normalisation constants are buffers so they travel with the weights and
    the numpy inference port reproduces them exactly.  `keep` selects input
    columns; `alphas` adds (EMA - x) features at those smoothing rates.
    """

    def __init__(self, mean, std, hidden=128, layers=2, d_in=128, keep=None, alphas=()):
        super().__init__()
        keep = list(range(N_FEATURES)) if keep is None else list(keep)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("keep", torch.as_tensor(keep, dtype=torch.long), persistent=False)
        self.alphas = list(alphas)
        self.inp = nn.Linear(len(keep) * (2 + len(self.alphas)), d_in)
        self.gru = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden + d_in, 2)
        self.hidden, self.layers = hidden, layers

    def normalise(self, x):
        return torch.clamp((x - self.mean) / self.std, -8, 8)[..., self.keep]

    def ema_features(self, x):
        """EMA deviations over a whole sequence x [B,T,112] raw; None when no alphas."""
        if not self.alphas:
            return None
        xn = self.normalise(x)
        return torch.cat([ema_scan(xn, a) - xn for a in self.alphas], dim=-1)

    def features(self, x, prev, ema=None):
        """x: [B,T,112] raw, prev: [B,112] raw value preceding x[:,0]."""
        xn, pn = self.normalise(x), self.normalise(prev)
        shifted = torch.cat([pn.unsqueeze(1), xn[:, :-1]], dim=1)
        parts = [xn, xn - shifted] + ([ema] if ema is not None else [])
        return torch.cat(parts, dim=-1)

    def forward(self, x, prev, h=None, ema=None):
        if self.alphas and ema is None:  # whole sequence given at once (evaluation)
            ema = self.ema_features(x)
        z = torch.relu(self.inp(self.features(x, prev, ema)))
        out, h = self.gru(z, h)
        return self.head(torch.cat([out, z], dim=-1)), h


def wpearson_loss(pred, y, mask):
    """Negative mean of the two weighted Pearson correlations (differentiable)."""
    yc = torch.clamp(y, -CLIP, CLIP)
    total = 0.0
    for k in range(2):
        w = yc[..., k].abs() * mask
        p, t = pred[..., k], yc[..., k]
        W = w.sum() + 1e-8
        mt, mp = (w * t).sum() / W, (w * p).sum() / W
        cov = (w * (t - mt) * (p - mp)).sum() / W
        vt = (w * (t - mt) ** 2).sum() / W
        vp = (w * (p - mp) ** 2).sum() / W
        total = total + cov / torch.sqrt(vt * vp + 1e-12)
    return -total / 2


class Accumulator:
    """Global weighted Pearson via pooled sums (float64), same result as utils.GlobalAccumulator."""

    def __init__(self):
        self.s = np.zeros((2, 6), dtype=np.float64)  # W, Wy, Wp, Wyy, Wpp, Wyp

    def add(self, y, p, mask):
        y = np.clip(y[mask], -CLIP, CLIP).astype(np.float64)
        p = np.clip(p[mask], -CLIP, CLIP).astype(np.float64)
        for k in range(2):
            w = np.abs(y[:, k])
            a, b = y[:, k], p[:, k]
            self.s[k] += [w.sum(), (w * a).sum(), (w * b).sum(),
                          (w * a * a).sum(), (w * b * b).sum(), (w * a * b).sum()]

    def result(self):
        r = []
        for W, Wy, Wp, Wyy, Wpp, Wyp in self.s:
            my, mp = Wy / W, Wp / W
            cov = Wyp / W - my * mp
            vy, vp = Wyy / W - my * my, Wpp / W - mp * mp
            r.append(cov / np.sqrt(vy * vp) if vy > 1e-16 and vp > 1e-16 else 0.0)
        return {"t0": r[0], "t1": r[1], "wp": float(np.mean(r))}
