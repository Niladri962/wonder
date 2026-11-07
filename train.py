"""Train the stateful GRU on train.parquet, select on valid.parquet.

Usage: python train.py --epochs 6 --hidden 128
"""
import argparse
import json
import time

import torch
import numpy as np
import pyarrow.parquet as pq

from common import (ART_DIR, CLIP, TRAIN_PATH, VALID_PATH, WARMUP, Accumulator, Net,
                    ThreadedBatches, read_sequence, wpearson_loss)
from mask_model import MaskNet, valid_halves

# From importance.py: trade prices (i0/i1 dp), a0-a3, a7 and i1 bid volume add nothing.
USELESS = set(range(44, 48)) | set(range(96, 100)) | {104, 105, 106, 107, 111} | set(range(74, 85))


def feature_stats(n_seq=200, seed=0):
    """Per-feature mean/std from a random sample of training sequences."""
    pf = pq.ParquetFile(str(TRAIN_PATH))
    rng = np.random.default_rng(seed)
    groups = rng.choice(pf.num_row_groups, n_seq, replace=False)
    xs = [read_sequence(pf, int(g))[0][::10] for g in groups]
    x = np.concatenate(xs).astype(np.float64)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


@torch.no_grad()
def evaluate(model, groups, device, batch=8):
    model.eval()
    pf = pq.ParquetFile(str(VALID_PATH))
    acc = Accumulator()
    for i in range(0, len(groups), batch):
        seqs = [read_sequence(pf, g, with_mask=True) for g in groups[i:i + batch]]
        x = torch.from_numpy(np.stack([s[0] for s in seqs])).to(device)
        pred, _ = model(x, x[:, 0])
        pred = pred.float().cpu().numpy()
        for (xs, ys, ms), p in zip(seqs, pred):
            ms[:WARMUP] = False
            acc.add(ys, p, ms)
    model.train()
    return acc.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-in", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-seqs", type=int, default=0, help="0 = all")
    ap.add_argument("--val-seqs", type=int, default=400, help="per-epoch subset; full set at the end")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default="gru")
    ap.add_argument("--mask-model", default="", help="artifacts/<name>.pt from mask_model.py; empty = uniform rows")
    ap.add_argument("--mask-floor", type=float, default=0.1, help="row weight = floor + P(scored)")
    ap.add_argument("--mse", type=float, default=0.0, help="weight of the |y|-weighted MSE scale anchor")
    ap.add_argument("--mask-power", type=float, default=1.0, help="row weight = floor + P(scored)**power")
    ap.add_argument("--add-valid-a", action="store_true", help="also train on validation half A")
    ap.add_argument("--drop-useless", action="store_true",
                    help="drop columns with ~zero permutation importance (dp, a0-a3, a7, i1 bid volume)")
    ap.add_argument("--alphas", default="", help="comma-separated EMA rates, e.g. 0.01,0.001")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ART_DIR.mkdir(exist_ok=True)

    t = time.time()
    mean, std = feature_stats()
    print(f"feature stats in {time.time() - t:.0f}s", flush=True)

    n_train = pq.ParquetFile(str(TRAIN_PATH)).num_row_groups
    n_valid = pq.ParquetFile(str(VALID_PATH)).num_row_groups
    train_groups = np.random.default_rng(1).permutation(n_train)
    if args.train_seqs:
        train_groups = train_groups[:args.train_seqs]
    # Select epochs on half B only: the mask model is fitted on half A.
    half_a, half_b = valid_halves()
    val_sub = list(np.random.default_rng(2).permutation(half_b)[:args.val_seqs])
    mask_net = None
    if args.mask_model:
        mask_net = MaskNet(np.zeros(112), np.ones(112)).to(device)
        mask_net.load_state_dict(torch.load(ART_DIR / f"{args.mask_model}.pt"))
        mask_net.eval().requires_grad_(False)

    args.keep = [i for i in range(112) if i not in USELESS] if args.drop_useless else None
    args.alphas = [float(a) for a in args.alphas.split(",") if a]
    model = Net(mean, std, args.hidden, args.layers, args.d_in, args.keep, args.alphas).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    items = [(TRAIN_PATH, g) for g in train_groups]
    if args.add_valid_a:
        items += [(VALID_PATH, g) for g in half_a]
    loader = ThreadedBatches(items, args.batch, threads=args.workers)
    steps_per_epoch = len(loader) * (20_000 // args.chunk)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05)

    best, cfg = -1.0, vars(args)
    for epoch in range(args.epochs):
        t, losses = time.time(), []
        for bi, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            h, prev = None, x[:, 0]
            with torch.no_grad():
                ema = model.ema_features(x)
            if mask_net is not None:
                with torch.no_grad():
                    row_w = args.mask_floor + torch.sigmoid(mask_net(x)) ** args.mask_power
            else:
                row_w = torch.ones(y.shape[:2], device=device)
            for c in range(0, x.shape[1], args.chunk):
                xc, yc = x[:, c:c + args.chunk], y[:, c:c + args.chunk]
                mask = row_w[:, c:c + args.chunk].clone()
                if c < WARMUP:
                    mask[:, :WARMUP - c] = 0
                pred, h = model(xc, prev, h, None if ema is None else ema[:, c:c + args.chunk])
                loss = wpearson_loss(pred, yc, mask)
                if args.mse:
                    t_clip = torch.clamp(yc, -CLIP, CLIP)
                    w = t_clip.abs() * mask.unsqueeze(-1)
                    loss = loss + args.mse * (w * (pred - t_clip) ** 2).sum() / (w.sum() + 1e-8)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                h, prev = h.detach(), xc[:, -1]
                losses.append(loss.item())
            if bi % 50 == 0:
                print(f"ep {epoch} batch {bi}/{len(loader)} loss {np.mean(losses[-200:]):.4f} "
                      f"{time.time() - t:.0f}s", flush=True)
        res = evaluate(model, val_sub, device)
        print(f"== epoch {epoch} train_loss {np.mean(losses):.4f} val(sub) {res} "
              f"{time.time() - t:.0f}s", flush=True)
        if res["wp"] > best:
            best = res["wp"]
            torch.save({"state": model.state_dict(), "cfg": cfg}, ART_DIR / f"{args.name}.pt")

    ckpt = torch.load(ART_DIR / f"{args.name}.pt")
    model.load_state_dict(ckpt["state"])
    full = evaluate(model, list(range(n_valid)), device)
    hb = evaluate(model, half_b, device)
    print(f"FULL VALID {full}", flush=True)
    print(f"HALF-B VALID (unseen by mask model) {hb}", flush=True)
    json.dump({"cfg": cfg, "full_valid": full, "half_b_valid": hb},
              open(ART_DIR / f"{args.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
