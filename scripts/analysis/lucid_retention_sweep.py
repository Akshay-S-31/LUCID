"""Measure ASM-gate data retention on URHI as a function of tau_ASM.

Streams real hazy images from URHI through the pretrained teacher, recomputes
the physical drift error exactly as optimize_parameters() does, blocks it at the
configured block size, and reports the fraction of blocks that pass the gate for
each candidate tau. Inference only -- no training, no images written to disk.

This is the retention quantity the gate actually sees at the start of any
fine-tuning run, since net_g_ema is initialised from CORUN+. It answers "which
tau values are even viable" before committing GPU time to a sweep.

Mirrors, rather than imports, the gate arithmetic from
colabator_by_depth_model.py (lines ~328-295) so the existing model code is left
untouched.

Usage:
    PYTHONPATH=.:basicsr_modified:corun_colabator python3 lucid_retention_sweep.py \
        --taus 0.02,0.05,0.10,0.15,0.20 --limit 500
"""

import argparse
import csv
import os
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


def load_corun(weights, device, depth=4):
    pkg = types.ModuleType('corun_colabator')
    pkg.__path__ = ['corun_colabator']
    sys.modules.setdefault('corun_colabator', pkg)
    sub = types.ModuleType('corun_colabator.archs')
    sub.__path__ = ['corun_colabator/archs']
    sys.modules.setdefault('corun_colabator.archs', sub)
    from corun_colabator.archs.corun_arch import CORUN

    net = CORUN(depth=depth)
    ckpt = torch.load(weights, map_location='cpu')
    key = 'params_ema' if 'params_ema' in ckpt else 'params'
    net.load_state_dict(ckpt[key], strict=True)
    net.to(device).eval()
    return net


def block_mean(x, bs):
    """Mean of each bs x bs block. Matches block_image(...).mean((1,2,3))."""
    B, C, H, W = x.shape
    H2, W2 = (H // bs) * bs, (W // bs) * bs
    if H2 == 0 or W2 == 0:
        return None
    x = x[:, :, :H2, :W2]
    x = x.reshape(B, C, H2 // bs, bs, W2 // bs, bs)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C * bs * bs).mean(dim=1)


def read_tensor(path, device, size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.)
    return t.unsqueeze(0).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='CORUN+.pth')
    p.add_argument('--urhi', default='Datasets/URHI')
    p.add_argument('--taus', default='0.02,0.05,0.10,0.15,0.20')
    p.add_argument('--block_size', type=int, default=32)
    p.add_argument('--size', type=int, default=288,
                   help='matches gt_size in the training config')
    p.add_argument('--limit', type=int, default=500)
    p.add_argument('--no_csv', action='store_true')
    args = p.parse_args()

    taus = [float(t) for t in args.taus.split(',')]
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}  block_size={args.block_size}  size={args.size}')

    files = []
    for root, _, fs in os.walk(args.urhi):
        for f in fs:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(root, f))
    files.sort()
    if args.limit:
        files = files[:args.limit]
    print(f'URHI images: {len(files)}')

    net = load_corun(args.weights, device)

    retained = {t: 0 for t in taus}
    total_blocks = 0
    errs = []
    failures = []

    for fp in tqdm(files, desc='URHI', unit='img'):
        try:
            real = read_tensor(fp, device, args.size)
            if real is None:
                failures.append(fp)
                continue
            with torch.no_grad():
                pl, pt = net(real, finetune=True)
                pseudo_label = pl[0].clamp(0, 1)
                pseudo_transmission = pt[0]
                # physical drift error, as in optimize_parameters
                I_recon = (pseudo_label * pseudo_transmission
                           + (1 - pseudo_transmission))
                phys_error = torch.mean((I_recon - real) ** 2, dim=1, keepdim=True)

                bm = block_mean(phys_error, args.block_size)
                if bm is None:
                    continue
                errs.append(bm.flatten().cpu())
                total_blocks += bm.numel()
                for t in taus:
                    retained[t] += (bm < t).sum().item()
            del real, pl, pt, pseudo_label, pseudo_transmission, I_recon, phys_error
        except Exception as exc:  # noqa: BLE001
            failures.append(f'{os.path.basename(fp)}: {type(exc).__name__}: {exc}')

    if not total_blocks:
        print('no blocks scored; aborting')
        return 1

    allerr = torch.cat(errs).numpy()
    print(f'\nphys_error per block over {total_blocks} blocks '
          f'({len(files)-len(failures)} images):')
    for q in (5, 25, 50, 75, 95):
        print(f'  p{q:<3d} {np.percentile(allerr, q):.5f}')
    print(f'  mean {allerr.mean():.5f}   max {allerr.max():.5f}')

    print('\ntau_ASM -> data retention (fraction of blocks passing the gate)')
    rows = []
    for t in taus:
        r = retained[t] / total_blocks
        flag = ''
        if r < 0.01:
            flag = '  [FLAG] near-zero retention: gate is closed, ' \
                   'fine-tuning would see almost no pseudo-label signal'
        elif r > 0.99:
            flag = '  [FLAG] saturated: gate is effectively disabled at this tau'
        print(f'  tau={t:<6.3f} retention={r:.4f}{flag}')
        rows.append((t, r, flag))

    if failures:
        print(f'\n[FLAG] {len(failures)} image(s) failed: {failures[:3]}')

    if not args.no_csv:
        os.makedirs('results', exist_ok=True)
        exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not exists:
                w.writeheader()
            for t, r, flag in rows:
                w.writerow({
                    'experiment_id': 'hp_tau_asm_retention' + ('  [FLAGGED]' if flag else ''),
                    'variant': 'gate_retention_at_pretrained_teacher',
                    'hyperparam_value': t,
                    'dataset': 'URHI',
                    'psnr': '', 'ssim': '', 'lpips': '', 'brisque': '', 'nima': '',
                    'best_checkpoint_iter': 'pretrained',
                    'data_retention_mean': f'{r:.4f}',
                })
        print(f'\nrows appended to {CSV_PATH}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
