"""Run training experiments one after another (the 4 GB GPU cannot hold two).

Usage: python run_queue.py [pid-to-wait-for] [run names...]   (default: all runs)
"""
import subprocess
import sys
import time
from pathlib import Path

WD = Path(__file__).resolve().parent
RUNS = dict([
    ("gru_v3", "--epochs 3 --batch 32 --lr 2e-3 --mask-model maskgru --mse 0.5 --add-valid-a"),
    ("gru_v4", "--epochs 3 --batch 32 --lr 2e-3 --mask-model maskgru --mse 0.5 --add-valid-a "
               "--mask-power 2 --mask-floor 0.05"),
    ("gru_v5", "--epochs 3 --batch 32 --lr 2e-3 --mask-model maskgru --mse 0.5 --add-valid-a "
               "--drop-useless --alphas 0.01,0.001"),
])


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def main():
    if len(sys.argv) > 1:
        while alive(int(sys.argv[1])):
            time.sleep(30)
    names = sys.argv[2:] or list(RUNS)
    for name in names:
        extra = RUNS[name]
        with open(WD / "logs" / f"{name}.log", "w") as out, open(WD / "logs" / f"{name}.err", "w") as err:
            subprocess.run([sys.executable, "-u", "train.py", *extra.split(), "--name", name],
                           cwd=WD, stdout=out, stderr=err)


if __name__ == "__main__":
    main()
