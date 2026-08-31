"""Compute FADE (Choi et al. 2015) over the RTTS set for the hazy inputs and
each dehazed result set. Uses PyFADE, which ships the original MATLAB MVG
reference models and is validated against the MATLAB reference to ~1e-6."""
import csv, json, os, sys, time
from pyfade import fade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fade_out")
os.makedirs(OUT, exist_ok=True)

SETS = [
    ("hazy_input",       f"{ROOT}/Datasets/RTTS/RTTS/JPEGImages"),
    ("lucid_5k",         f"{ROOT}/results/Best_BRISQUE_5k/RTTS"),
    ("corun_plus_base",  f"{ROOT}/results/Baseline_CORunPlus/RTTS"),
    ("lucid_30k",        f"{ROOT}/results/Best_NIMA_30k/RTTS"),
]

summary = {}
for name, path in SETS:
    t = time.time()
    res = fade(path, workers=8, progress=False)
    scores = dict(res.scores)
    with open(f"{OUT}/fade_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "fade"])
        for k in sorted(scores):
            w.writerow([k, "%.12f" % scores[k]])
    summary[name] = {
        "dir": path, "n": len(scores),
        "mean": float(res.mean_score),
        "min": float(res.min_score), "max": float(res.max_score),
        "seconds": round(time.time() - t, 1),
    }
    print(f"{name:16s} n={len(scores):5d}  mean FADE={res.mean_score:.6f}  "
          f"[{res.min_score:.4f}, {res.max_score:.4f}]  ({time.time()-t:.0f}s)", flush=True)

with open(f"{OUT}/fade_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nDONE")
