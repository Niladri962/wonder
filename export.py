"""Export a trained checkpoint to submission/model.npz and check numpy == torch.

Usage: python export.py --name gru
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

import torch
import numpy as np
import pyarrow.parquet as pq

from common import ART_DIR, DATA_ROOT, VALID_PATH, Net, read_sequence

SUB_DIR = Path(__file__).resolve().parent / "submission"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gru")
    args = ap.parse_args()

    ckpt = torch.load(ART_DIR / f"{args.name}.pt", map_location="cpu")
    cfg, sd = ckpt["cfg"], ckpt["state"]
    model = Net(sd["mean"], sd["std"], cfg["hidden"], cfg["layers"], cfg["d_in"],
                cfg.get("keep"), cfg.get("alphas", ()))
    model.load_state_dict(sd)
    model.eval()

    arrays = {"mean": sd["mean"], "std": sd["std"], "inp_w": sd["inp.weight"], "inp_b": sd["inp.bias"],
              "head_w": sd["head.weight"], "head_b": sd["head.bias"]}
    for l in range(cfg["layers"]):
        for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            short = {"weight_ih": "w_ih", "weight_hh": "w_hh", "bias_ih": "b_ih", "bias_hh": "b_hh"}[k]
            arrays[f"{short}{l}"] = sd[f"gru.{k}_l{l}"]
    arrays = {k: v.numpy().astype(np.float32) for k, v in arrays.items()}
    arrays["layers"], arrays["hidden"] = np.array(cfg["layers"]), np.array(cfg["hidden"])
    if cfg.get("keep") is not None:
        arrays["keep"] = np.array(cfg["keep"], dtype=np.int64)
    if cfg.get("alphas"):
        arrays["alphas"] = np.array(cfg["alphas"], dtype=np.float32)
    np.savez(SUB_DIR / "model.npz", **arrays)

    # Parity check on one validation sequence: batched torch vs step-by-step numpy.
    sys.path.insert(0, str(SUB_DIR))
    sys.path.insert(0, str(DATA_ROOT))
    from solution import PredictionModel
    from utils import DataPoint
    x, y = read_sequence(pq.ParquetFile(str(VALID_PATH)), 0)
    n = 3000
    with torch.no_grad():
        xt = torch.from_numpy(x[None, :n])
        ref = model(xt, xt[:, 0])[0][0].numpy()
    pm = PredictionModel()
    t = time.perf_counter()
    out = [pm.predict(DataPoint(0, i, i >= 99, x[i])) for i in range(n)]
    per_step = (time.perf_counter() - t) / n * 1e6
    got = np.stack(out[99:])
    print(f"max |numpy - torch| = {np.abs(got - ref[99:]).max():.2e}, {per_step:.1f} us/step "
          f"-> ~{per_step * 39.4e6 / 1e6 / 60:.1f} min for the 39.4M-row test set")

    zpath = SUB_DIR.parent / f"submission_{args.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(SUB_DIR / "solution.py", "solution.py")
        z.write(SUB_DIR / "model.npz", "model.npz")
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
