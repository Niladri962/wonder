"""Score a saved checkpoint on the full validation set and on half B.

Usage: python evaluate.py gru_mask
"""
import json
import sys

import torch

from common import ART_DIR
from diag import load
from mask_model import valid_halves
from train import evaluate


def main():
    name = sys.argv[1]
    device = torch.device("cuda")
    model = load(name, device)
    full = evaluate(model, list(range(1873)), device)
    hb = evaluate(model, valid_halves()[1], device)
    print(f"{name} FULL VALID {full}\n{name} HALF-B VALID {hb}", flush=True)
    json.dump({"full_valid": full, "half_b_valid": hb}, open(ART_DIR / f"{name}.eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
