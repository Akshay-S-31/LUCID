"""Streaming, in-memory evaluation for LUCID / CORUN+ checkpoints.

Loads one test image at a time, runs the model, computes metrics against the
decoded tensors, and discards the output. No dehazed image is ever written to
disk -- not as PNG, not as a temporary file. Only numeric rows are persisted,
appended to results/experiments.csv.

Paired sets (SOTS-Indoor)  -> PSNR, SSIM, LPIPS
Unpaired sets (RTTS)       -> BRISQUE, NIMA

Usage:
    PYTHONPATH=.:basicsr_modified:corun_colabator python3 lucid_stream_eval.py \
        --weights CORUN+.pth --experiment_id baseline --variant pretrained_corun_plus

The PYTHONPATH needs the third entry because corun_colabator/data does a bare
`from utils import ...`; see also evaluate_full.py.
"""

import argparse
import csv
import os
import shutil
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault('HF_HUB_OFFLINE', 'True')

CSV_PATH = os.path.join('results', 'experiments.csv')
CSV_COLUMNS = [
    'experiment_id', 'variant', 'hyperparam_value', 'dataset',
    'psnr', 'ssim', 'lpips', 'brisque', 'nima',
    'best_checkpoint_iter', 'data_retention_mean',
]

# PSNR outside this band on a dehazing benchmark means something is broken
# (identity passthrough, channel swap, wrong GT pairing) rather than a real result.
PSNR_SANE = (5.0, 50.0)


def load_corun(weights, device, depth=4):
    """Import CORUN without triggering the package __init__ side effects."""
    pkg = types.ModuleType('corun_colabator')
    pkg.__path__ = ['corun_colabator']
    sys.modules.setdefault('corun_colabator', pkg)
    sub = types.ModuleType('corun_colabator.archs')
    sub.__path__ = ['corun_colabator/archs']
    sys.modules.setdefault('corun_colabator.archs', sub)
    from corun_colabator.archs.corun_arch import CORUN

    net = CORUN(depth=depth)
    ckpt = torch.load(weights, map_location='cpu')
    key = 'params_ema' if 'params_ema' in ckpt else ('params' if 'params' in ckpt else None)
    state = ckpt[key] if key else ckpt
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    return net


def read_tensor(path, device, size=None):
    """BGR uint8 file -> RGB float tensor in [0,1], shape [1,3,H,W]. Stays in RAM."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.)
    return t.unsqueeze(0).to(device)


def pad_to(t, factor=8):
    h, w = t.shape[-2:]
    H = ((h + factor - 1) // factor) * factor
    W = ((w + factor - 1) // factor) * factor
    return F.pad(t, (0, W - w, 0, H - h), mode='reflect'), h, w


@torch.no_grad()
def dehaze(net, lq):
    padded, h, w = pad_to(lq)
    outs = net(padded, debug=False)
    out = outs[0] if isinstance(outs, (tuple, list)) else outs
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out[..., :h, :w].clamp(0, 1)


def disk_used_gb(path='.'):
    total = 0
    for root, _, files in os.walk(path):
        if '.git' in root:
            continue
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


class DiskGuard:
    """Halt loudly if this process grows the working tree past the budget."""

    def __init__(self, budget_gb, root='.'):
        self.budget = budget_gb
        self.root = root
        self.baseline = disk_used_gb(root)

    def check(self):
        free_gb = shutil.disk_usage(self.root).free / 1e9
        grown = disk_used_gb(self.root) - self.baseline
        if grown > self.budget:
            raise RuntimeError(
                f'HALT: working tree grew {grown:.2f} GB this run, over the '
                f'{self.budget:.0f} GB budget. Nothing further will be written.')
        if free_gb < 5.0:
            raise RuntimeError(f'HALT: only {free_gb:.1f} GB free on the volume.')
        return grown


def evaluate(net, device, dataset, hazy_dir, gt_dir, size, metrics, limit=None,
             guard=None):
    """Stream over a directory, accumulating metric sums. Outputs are discarded."""
    files = sorted(f for f in os.listdir(hazy_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if limit:
        files = files[:limit]

    sums, counts = {k: 0.0 for k in metrics}, {k: 0 for k in metrics}
    nan_hits, missing_gt, failures = [], 0, []

    for i, fname in enumerate(tqdm(files, desc=dataset, unit='img')):
        try:
            lq = read_tensor(os.path.join(hazy_dir, fname), device, size)
            if lq is None:
                failures.append(fname)
                continue

            out = dehaze(net, lq)

            gt = None
            if gt_dir:
                gt_path = os.path.join(gt_dir, fname)
                if not os.path.exists(gt_path):
                    stem = os.path.splitext(fname)[0].split('_')[0]
                    for ext in ('.png', '.jpg', '.jpeg'):
                        cand = os.path.join(gt_dir, stem + ext)
                        if os.path.exists(cand):
                            gt_path = cand
                            break
                if os.path.exists(gt_path):
                    gt = read_tensor(gt_path, device, size)
                    if gt is not None and gt.shape[-2:] != out.shape[-2:]:
                        gt = F.interpolate(gt, size=out.shape[-2:],
                                           mode='bilinear', align_corners=False)
                else:
                    missing_gt += 1

            for name, (fn, needs_gt) in metrics.items():
                if needs_gt and gt is None:
                    continue
                try:
                    with torch.no_grad():
                        val = fn(out, gt) if needs_gt else fn(out)
                except Exception as exc:  # noqa: BLE001 - isolate per metric
                    failures.append(f'{fname}[{name}]: {type(exc).__name__}: {exc}')
                    continue
                if not np.isfinite(val):
                    nan_hits.append((fname, name))
                    continue
                sums[name] += val
                counts[name] += 1

            # explicit discard: nothing from this image survives the iteration
            del lq, out, gt

        except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
            failures.append(f'{fname}: {type(exc).__name__}: {exc}')

        if guard and i % 200 == 0:
            guard.check()

    means = {k: (sums[k] / counts[k] if counts[k] else float('nan')) for k in metrics}
    return means, {
        'n': len(files), 'counts': counts, 'nan': nan_hits,
        'missing_gt': missing_gt, 'failures': failures,
    }


def flag_row(dataset, means, diag):
    """Return a list of human-readable warnings; empty means the row looks sane."""
    warns = []
    if diag['failures']:
        warns.append(f"{len(diag['failures'])} image(s) failed: {diag['failures'][:3]}")
    if diag['nan']:
        warns.append(f"{len(diag['nan'])} non-finite metric value(s), e.g. {diag['nan'][:3]}")
    if diag['missing_gt']:
        warns.append(f"{diag['missing_gt']} image(s) had no matching ground truth")
    psnr = means.get('psnr')
    if psnr is not None and np.isfinite(psnr) and not (PSNR_SANE[0] <= psnr <= PSNR_SANE[1]):
        warns.append(f'PSNR {psnr:.2f} dB is outside the plausible band {PSNR_SANE}')
    ssim = means.get('ssim')
    if ssim is not None and np.isfinite(ssim) and not (0.0 <= ssim <= 1.0):
        warns.append(f'SSIM {ssim:.3f} outside [0,1]')
    for k, v in means.items():
        if not np.isfinite(v) and diag['counts'].get(k, 0) == 0:
            warns.append(f'{k}: no valid samples')
    return warns


def append_csv(row):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({c: row.get(c, '') for c in CSV_COLUMNS})


def main():
    p = argparse.ArgumentParser(description='In-memory streaming eval for LUCID')
    p.add_argument('--weights', default='CORUN+.pth')
    p.add_argument('--experiment_id', required=True)
    p.add_argument('--variant', default='')
    p.add_argument('--hyperparam_value', default='')
    p.add_argument('--best_checkpoint_iter', default='')
    p.add_argument('--data_retention_mean', default='')
    p.add_argument('--size', type=int, default=256,
                   help='square resize; 256 matches the existing simple_test.py pipeline')
    p.add_argument('--limit', type=int, default=None, help='cap images per set (smoke tests)')
    p.add_argument('--datasets', default='sots_indoor,rtts')
    p.add_argument('--disk_budget_gb', type=float, default=20.0)
    p.add_argument('--no_csv', action='store_true', help='print only, do not append to CSV')
    args = p.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    guard = DiskGuard(args.disk_budget_gb)
    net = load_corun(args.weights, device)
    print(f'loaded {args.weights}')

    import pyiqa
    made = {}

    def get(name):
        """Metric callable that falls back to CPU permanently if the accelerator
        backend cannot run it (pyiqa's SSIM needs float64, unsupported on MPS)."""
        if name in made:
            return made[name]

        state = {'m': pyiqa.create_metric(name, device=device), 'dev': device}

        def call(*tensors):
            args = [t.to(state['dev']) for t in tensors if t is not None]
            try:
                return state['m'](*args).item()
            except (TypeError, RuntimeError) as exc:
                if state['dev'].type == 'cpu':
                    raise
                print(f'  [info] {name} unsupported on {state["dev"].type} '
                      f'({type(exc).__name__}); falling back to CPU for this metric')
                state['m'] = pyiqa.create_metric(name, device=torch.device('cpu'))
                state['dev'] = torch.device('cpu')
                return state['m'](*[t.cpu() for t in tensors if t is not None]).item()

        made[name] = call
        return call

    specs = {
        'sots_indoor': dict(
            hazy='Datasets/SOTS/indoor/hazy',
            gt='Datasets/SOTS/indoor/gt_aligned',
            metrics=['psnr', 'ssim', 'lpips'],
        ),
        'rtts': dict(
            hazy='Datasets/RTTS/RTTS/JPEGImages',
            gt=None,
            metrics=['brisque', 'nima'],
        ),
    }

    all_warns = {}
    for ds in [d.strip() for d in args.datasets.split(',') if d.strip()]:
        spec = specs[ds]
        if not os.path.isdir(spec['hazy']):
            print(f'SKIP {ds}: {spec["hazy"]} not found')
            continue

        needs_gt = {'psnr', 'ssim', 'lpips'}
        metrics = {m: (get(m), m in needs_gt) for m in spec['metrics']}

        means, diag = evaluate(net, device, ds, spec['hazy'], spec['gt'],
                               args.size, metrics, args.limit, guard)

        warns = flag_row(ds, means, diag)
        all_warns[ds] = warns

        print(f'\n=== {ds} ({diag["n"]} images) ===')
        for k, v in means.items():
            print(f'  {k:8s} {v:.4f}  (n={diag["counts"][k]})')
        for w in warns:
            print(f'  [FLAG] {w}')

        if not args.no_csv:
            row = {
                'experiment_id': args.experiment_id,
                'variant': args.variant,
                'hyperparam_value': args.hyperparam_value,
                'dataset': ds,
                'best_checkpoint_iter': args.best_checkpoint_iter,
                'data_retention_mean': args.data_retention_mean,
            }
            row.update({k: f'{v:.4f}' for k, v in means.items()})
            if warns:
                row['experiment_id'] = args.experiment_id + '  [FLAGGED]'
            append_csv(row)

    grown = guard.check()
    print(f'\nworking tree grew {grown*1000:.1f} MB this run (budget '
          f'{args.disk_budget_gb:.0f} GB); zero images written.')
    if not args.no_csv:
        print(f'rows appended to {CSV_PATH}')
    if any(all_warns.values()):
        print('\nSome rows were FLAGGED above -- treat them as unreliable.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
