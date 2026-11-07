"""Stateful GRU, run step by step in pure numpy (single thread, deterministic).

Per layer one fused matmul: [x, h] @ W -> [rz_pre, i_n, h_n], so each step is
just a handful of numpy calls.
"""
from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


class PredictionModel:
    def __init__(self):
        w = np.load(Path(__file__).resolve().parent / "model.npz")
        keep = w["keep"] if "keep" in w else np.arange(w["mean"].shape[0])
        self.mean, self.inv_std = w["mean"][keep], (1.0 / w["std"][keep]).astype(np.float32)
        self.keep = keep
        self.alphas = [np.float32(a) for a in w["alphas"]] if "alphas" in w else []
        self.w_inp, self.b_inp = w["inp_w"].T.copy(), w["inp_b"]
        self.layers, H = int(w["layers"]), int(w["hidden"])
        self.H = H
        self.w, self.b = [], []
        for l in range(self.layers):
            w_ih, w_hh = w[f"w_ih{l}"].T, w[f"w_hh{l}"].T  # [in,3H], [H,3H]
            b_ih, b_hh = w[f"b_ih{l}"], w[f"b_hh{l}"]
            d = w_ih.shape[0]
            fused = np.zeros((d + H, 4 * H), dtype=np.float32)
            fused[:d, :2 * H] = w_ih[:, :2 * H]
            fused[d:, :2 * H] = w_hh[:, :2 * H]
            fused[:d, 2 * H:3 * H] = w_ih[:, 2 * H:]
            fused[d:, 3 * H:] = w_hh[:, 2 * H:]
            bias = np.concatenate([b_ih[:2 * H] + b_hh[:2 * H], b_ih[2 * H:], b_hh[2 * H:]])
            # sigmoid(v) = 0.5 * tanh(v / 2) + 0.5: pre-halve the r/z columns.
            fused[:, :2 * H] *= 0.5
            bias[:2 * H] *= 0.5
            self.w.append(fused)
            self.b.append(bias.astype(np.float32))
        self.w_head, self.b_head = w["head_w"].T.copy(), w["head_b"]
        # Layout: [x_norm | x_norm - prev | (ema_k - x_norm) for each alpha]
        self.xbuf = np.zeros((2 + len(self.alphas)) * self.mean.shape[0], dtype=np.float32)
        self.seq_ix = None

    def predict(self, data_point):
        n_f = self.mean.shape[0]
        xbuf = self.xbuf
        state = np.asarray(data_point.state, dtype=np.float32)[self.keep]
        xn = np.clip((state - self.mean) * self.inv_std, -8.0, 8.0)
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.h = [np.zeros(self.H, dtype=np.float32) for _ in range(self.layers)]
            self.ema = [xn.copy() for _ in self.alphas]
            xbuf[:n_f] = xn
        # xbuf[:n_f] holds the previous normalised row.
        xbuf[n_f:2 * n_f] = xn - xbuf[:n_f]
        xbuf[:n_f] = xn
        for k, a in enumerate(self.alphas):
            e = self.ema[k]
            e += a * (xn - e)
            xbuf[(2 + k) * n_f:(3 + k) * n_f] = e - xn
        z = np.maximum(xbuf @ self.w_inp + self.b_inp, 0.0)
        H, inp = self.H, z
        for l in range(self.layers):
            g = np.concatenate([inp, self.h[l]]) @ self.w[l] + self.b[l]
            rz = np.tanh(g[:2 * H]) * 0.5 + 0.5
            n = np.tanh(g[2 * H:3 * H] + rz[:H] * g[3 * H:])
            u = rz[H:]
            inp = self.h[l] = n + u * (self.h[l] - n)
        if not data_point.need_prediction:
            return None
        return np.concatenate([inp, z]) @ self.w_head + self.b_head


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    parser.add_argument("--utils-dir", required=True, help="starter pack root containing utils.py")
    args = parser.parse_args()
    sys.path.insert(0, args.utils_dir)
    from utils import ScorerStepByStep
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
