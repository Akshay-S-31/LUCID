import csv, itertools, os
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fade_out")
names = ["hazy_input", "lucid_5k", "corun_plus_base", "lucid_30k"]

def load(n):
    d = {}
    with open(f"{OUT}/fade_{n}.csv") as f:
        for row in csv.DictReader(f):
            d[row["image"]] = float(row["fade"])
    return d

data = {n: load(n) for n in names}
keys = sorted(set.intersection(*[set(d) for d in data.values()]))
print(f"paired images: {len(keys)}\n")
arr = {n: np.array([data[n][k] for k in keys]) for n in names}

print(f"{'set':16s} {'mean':>9s} {'median':>9s} {'std':>8s}")
for n in names:
    a = arr[n]
    print(f"{n:16s} {a.mean():9.4f} {np.median(a):9.4f} {a.std(ddof=1):8.4f}")

print("\npaired comparisons (lower FADE = less residual haze):")
pairs = [("lucid_5k", "corun_plus_base"), ("lucid_30k", "corun_plus_base"), ("lucid_30k", "lucid_5k")]
for a_n, b_n in pairs:
    a, b = arr[a_n], arr[b_n]
    d = a - b                      # negative => a better
    win = float((d < 0).mean()) * 100
    w = stats.wilcoxon(a, b)
    t = stats.ttest_rel(a, b)
    coh = d.mean() / d.std(ddof=1)
    print(f"\n  {a_n} vs {b_n}")
    print(f"    mean diff      {d.mean():+.4f}  ({100*d.mean()/b.mean():+.2f}% vs {b_n})")
    print(f"    median diff    {np.median(d):+.4f}")
    print(f"    {a_n} wins on  {win:.1f}% of images ({int((d<0).sum())}/{len(d)})")
    print(f"    Wilcoxon       W={w.statistic:.0f}  p={w.pvalue:.3e}")
    print(f"    paired t-test  t={t.statistic:.2f}  p={t.pvalue:.3e}   Cohen's dz={coh:+.3f}")

print("\nhaze reduction vs input (mean over images of 1 - out/in):")
for n in ["lucid_5k", "corun_plus_base", "lucid_30k"]:
    r = (1 - arr[n] / arr["hazy_input"]) * 100
    print(f"  {n:16s} {r.mean():6.2f}%")
