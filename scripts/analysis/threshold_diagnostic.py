"""Measure where the router's thresholds actually need to sit.

The ablation grid is only informative if tau_Q and tau_crit fall inside the
range where the mechanism changes behaviour. Both are compared against *block
averages* -- tau_Q against the mean U_joint of a 64x64 root block, tau_crit
against the mean of a leaf -- and block averages of a min-max normalised map
concentrate well below its pixel maximum, so a threshold chosen by reasoning
about the [0,1] range alone can turn out to be unreachable.

This script runs the pretrained teacher over real URHI crops exactly as
training does (S stochastic MC-Dropout passes, then the ASM reconstruction),
and reports the percentiles of

    * block-mean U_joint at each quadtree level (64, 32, 16, 8)
    * block-mean raw eps_phys at the gate's block size

From those percentiles you can read off thresholds that retain, say, 90% / 70%
/ 50% of blocks, and build a sweep that spans real behaviour.

Inference only. No training, no checkpoints, nothing written to disk.

Usage:
    PYTHONPATH=.:basicsr_modified:corun_colabator \
        python3 scripts/analysis/threshold_diagnostic.py --num 64
"""

import argparse
import os
import random
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault('HF_HUB_OFFLINE', 'True')

PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def load_corun(weights, device, depth=4, mc_dropout_p=0.1):
    """Import CORUN without triggering the package __init__ side effects."""
    pkg = types.ModuleType('corun_colabator')
    pkg.__path__ = ['corun_colabator']
    sys.modules.setdefault('corun_colabator', pkg)
    sub = types.ModuleType('corun_colabator.archs')
    sub.__path__ = ['corun_colabator/archs']
    sys.modules.setdefault('corun_colabator.archs', sub)
    from corun_colabator.archs.corun_arch import CORUN

    net = CORUN(depth=depth, mc_dropout_p=mc_dropout_p)
    ckpt = torch.load(weights, map_location='cpu')
    key = 'params_ema' if 'params_ema' in ckpt else ('params' if 'params' in ckpt else None)
    net.load_state_dict(ckpt[key] if key else ckpt, strict=True)
    net.to(device).eval()
    return net


def crops(hazy_dir, n, size, seed=0, mode='resize'):
    """Prepare real hazy inputs at `size` x `size`.

    mode='resize' reproduces what training actually feeds the teacher. With
    `use_resize_crop: true`, SemiHazeOnlineDataset does
    `cv2.resize(img_real, (gt_size, gt_size), INTER_CUBIC)` -- the whole photo
    is squashed to the crop size, aspect ratio and all. That downsampling
    smooths away high-frequency content, which tightens the spread of block
    means and so changes where the thresholds bite. It is the mode to calibrate
    against.

    mode='crop' takes a native-resolution window instead. Useful for seeing the
    signal without the downsampling, but it is not what training sees and will
    suggest thresholds that are too high.
    """
    files = sorted(f for f in os.listdir(hazy_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    rng = random.Random(seed)
    rng.shuffle(files)
    out = []
    for f in files:
        if len(out) >= n:
            break
        img = cv2.imread(os.path.join(hazy_dir, f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if mode == 'resize':
            patch = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        else:
            h, w = img.shape[:2]
            if h < size or w < size:
                img = cv2.resize(img, (max(size, w), max(size, h)))
                h, w = img.shape[:2]
            top, left = rng.randint(0, h - size), rng.randint(0, w - size)
            patch = img[top:top + size, left:left + size]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1).float().div_(255.)
        out.append(t.unsqueeze(0))
    return out


def block_means(m, block):
    """Mean of every non-overlapping block x block tile of a [1,1,H,W] map."""
    return F.avg_pool2d(m, kernel_size=block, stride=block).reshape(-1)


def report(name, values, note=''):
    v = np.sort(np.concatenate(values))
    qs = np.percentile(v, PCTS)
    print(f'\n{name}   n={v.size}{note}')
    print('  ' + '  '.join(f'p{p}={q:.4f}' for p, q in zip(PCTS, qs)))
    print(f'  min={v.min():.4f}  mean={v.mean():.4f}  max={v.max():.4f}')
    return v


def suggest(v, label, retentions=(0.90, 0.70, 0.50)):
    """Thresholds that would keep the given fractions of blocks."""
    print(f'  thresholds for {label}:')
    for r in retentions:
        # keep fraction r  <=>  threshold at the r-th quantile
        print(f'    retain {int(r * 100):>2d}%  ->  {np.quantile(v, r):.4f}')


def main():
    p = argparse.ArgumentParser(description='Router threshold diagnostic')
    p.add_argument('--weights', default='CORUN+.pth')
    p.add_argument('--real_dir', default='Datasets/URHI')
    p.add_argument('--num', type=int, default=64, help='number of crops')
    p.add_argument('--size', type=int, default=192, help='crop size (training gt_size)')
    p.add_argument('--mc_K', type=int, default=5)
    p.add_argument('--block_size', type=int, default=32, help='ASM gate block size')
    p.add_argument('--quadtree_sizes', default='64,32,16,8')
    p.add_argument('--simulate_router', action='store_true',
                   help='also run the real QuadtreeRouter and report, per crop, '
                        'how many leaves land at each level and how many are '
                        'discarded by tau_crit')
    p.add_argument('--tau_q', type=float, default=0.12)
    p.add_argument('--tau_crit', type=float, default=0.22)
    p.add_argument('--input_mode', choices=['resize', 'crop'], default='resize',
                   help="how to prepare real inputs: 'resize' matches training "
                        "(whole photo squashed to gt_size); 'crop' takes a "
                        "native-resolution window")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'device: {device}   inputs: {args.num} @ {args.size}x{args.size}   '
          f'S={args.mc_K}   mode={args.input_mode}'
          + ('  (matches training)' if args.input_mode == 'resize' else '  (NOT what training sees)'))

    from corun_colabator.archs.corun_arch import enable_mc_dropout, disable_mc_dropout
    from corun_colabator.archs.quadtree_router import joint_uncertainty

    net = load_corun(args.weights, device)
    sizes = [int(s) for s in args.quadtree_sizes.split(',')]

    u_by_size = {s: [] for s in sizes}
    gate_blocks = []
    phys_px = []

    for x in crops(args.real_dir, args.num, args.size, mode=args.input_mode):
        x = x.to(device)
        with torch.no_grad():
            enable_mc_dropout(net)
            try:
                js, ts = [], []
                for _ in range(args.mc_K):
                    pl, pt = net(x, finetune=True)
                    js.append(pl[0].clamp(0, 1))
                    ts.append(pt[0])
            finally:
                disable_mc_dropout(net)
            J = torch.stack(js, 0)
            T = torch.stack(ts, 0)
            j_mu, t_mu = J.mean(0), T.mean(0)
            var_j, var_t = J.var(0, unbiased=False), T.var(0, unbiased=False)

            recon = j_mu * t_mu + (1 - t_mu)
            eps = torch.mean((recon - x) ** 2, dim=1, keepdim=True)
            u = joint_uncertainty(var_j, var_t, eps)

        phys_px.append(eps.reshape(-1).cpu().numpy())
        gate_blocks.append(block_means(eps, args.block_size).cpu().numpy())
        for s in sizes:
            if args.size >= s:
                u_by_size[s].append(block_means(u, s).cpu().numpy())

    print('\n' + '=' * 72)
    print('ASM GATE -- raw eps_phys, the quantity tau_ASM is compared against')
    print('=' * 72)
    report('eps_phys per pixel', phys_px)
    v = report(f'eps_phys per {args.block_size}x{args.block_size} block', gate_blocks,
               note='   <- tau_ASM acts here')
    suggest(v, 'tau_ASM')

    print('\n' + '=' * 72)
    print('ROUTER -- normalised U_joint, what tau_Q and tau_crit are compared against')
    print('=' * 72)
    for s in sizes:
        if not u_by_size[s]:
            continue
        note = '   <- tau_Q acts here (root level)' if s == sizes[0] else ''
        v = report(f'U_joint per {s}x{s} block', u_by_size[s], note=note)
        if s == sizes[0]:
            suggest(v, 'tau_Q (fraction of root blocks kept whole)')
        if s == sizes[-1]:
            suggest(v, 'tau_crit (fraction of finest leaves kept)')

    print('\nRead the suggested values off the rows above and use them as the '
          'ablation grid;\na threshold beyond the p99 of its distribution will '
          'never fire.')

    if args.simulate_router:
        simulate(net, args, sizes, device)


def simulate(net, args, sizes, device):
    """Run the real router and report where leaves land and what gets discarded.

    Percentiles of block means only bound what *could* happen. Which blocks
    actually reach the finest level depends on the recursion: a block is only
    examined at 8x8 if every ancestor exceeded tau_Q, so the population seen at
    each level is a biased subset of the overall distribution rather than a
    sample from it. This runs the router itself so the leaf counts and the
    discard count are the real ones.
    """
    from corun_colabator.archs.corun_arch import enable_mc_dropout, disable_mc_dropout
    from corun_colabator.archs.quadtree_router import QuadtreeRouter, joint_uncertainty

    def walk(u, H, W):
        """Replicate QuadtreeRouter's traversal, returning (size, mean) leaves.

        The router does not expose the block means behind its decisions, and
        they are the quantity tau_crit is judged against. Note these are *not*
        a sample of all blocks at a level: a block is only examined at 8x8 if
        every ancestor exceeded tau_Q, so this population is selected by the
        recursion and is what tau_crit actually sees.
        """
        pending = [(t, l, sizes[0]) for t in range(0, H, sizes[0])
                   for l in range(0, W, sizes[0])]
        out = []
        for level, sz in enumerate(sizes):
            if not pending:
                break
            finest = (level == len(sizes) - 1)
            nxt = []
            for (top, left, s) in pending:
                bot, right = min(top + s, H), min(left + s, W)
                if bot <= top or right <= left:
                    continue
                bu = float(u[0, :, top:bot, left:right].mean())
                if bu <= args.tau_q or finest:
                    out.append((s, bu))
                else:
                    ch = sizes[level + 1]
                    for dy in range(0, s, ch):
                        for dx in range(0, s, ch):
                            if top + dy < H and left + dx < W:
                                nxt.append((top + dy, left + dx, ch))
            pending = nxt
        return out

    seen = {'by_size': {}}
    leaf_u = {s: [] for s in sizes}

    def scorer(x):
        # Constant 1.0: the mask then reflects only the router's hard discards,
        # isolating tau_crit from the DA-CLIP / MUSIQ weighting.
        seen['by_size'][x.shape[-1]] = seen['by_size'].get(x.shape[-1], 0) + x.shape[0]
        return torch.ones(x.shape[0], device=x.device)

    router = QuadtreeRouter(scorer, sizes=sizes,
                            tau_q=args.tau_q, tau_crit=args.tau_crit)
    retentions = []

    for x in crops(args.real_dir, min(args.num, 16), args.size, mode=args.input_mode):
        x = x.to(device)
        with torch.no_grad():
            enable_mc_dropout(net)
            try:
                js, ts = [], []
                for _ in range(args.mc_K):
                    pl, pt = net(x, finetune=True)
                    js.append(pl[0].clamp(0, 1))
                    ts.append(pt[0])
            finally:
                disable_mc_dropout(net)
            J, T = torch.stack(js, 0), torch.stack(ts, 0)
            j_mu, t_mu = J.mean(0), T.mean(0)
            var_j, var_t = J.var(0, unbiased=False), T.var(0, unbiased=False)
            eps = torch.mean((j_mu * t_mu + (1 - t_mu) - x) ** 2, dim=1, keepdim=True)
            u = joint_uncertainty(var_j, var_t, eps)
            m = router(j_mu, u)
        retentions.append((m > 0).float().mean().item())
        for sz, bu in walk(u, args.size, args.size):
            leaf_u[sz].append(bu)

    print('\n' + '=' * 72)
    print(f'ROUTER SIMULATION   tau_Q={args.tau_q}  tau_crit={args.tau_crit}')
    print('=' * 72)
    print('  variance magnitudes (before normalisation):')
    print(f'    var_J mean {var_j.mean().item():.3e}   max {var_j.max().item():.3e}')
    print(f'    var_T mean {var_t.mean().item():.3e}   max {var_t.max().item():.3e}')
    print(f'    eps   mean {eps.mean().item():.3e}   max {eps.max().item():.3e}')
    print(f'  U_joint: mean {u.mean().item():.4f}  max {u.max().item():.4f}')
    print('  leaves scored per level:')
    for s in sizes:
        print(f'    {s:>3d}x{s:<3d}  {seen["by_size"].get(s, 0)}')
    r = np.array(retentions)
    print(f'  data_retention: mean {r.mean():.4f}  min {r.min():.4f}  max {r.max():.4f}')
    print('\n  leaf block-mean U_joint, as selected by the recursion')
    print('  (this, not the distribution over all blocks, is what tau_crit sees):')
    allv = []
    for s in sizes:
        v = np.array(leaf_u[s])
        if v.size == 0:
            continue
        allv.append(v)
        qs = np.percentile(v, [50, 90, 95, 99])
        print(f'    {s:>3d}x{s:<3d} n={v.size:<6d} '
              f'p50={qs[0]:.4f} p90={qs[1]:.4f} p95={qs[2]:.4f} p99={qs[3]:.4f} '
              f'max={v.max():.4f}')
    if allv:
        v = np.concatenate(allv)
        print(f'\n  across all leaves: max={v.max():.4f}')
        print(f'  tau_crit that would discard  1% of leaves: {np.quantile(v, 0.99):.4f}')
        print(f'  tau_crit that would discard  5% of leaves: {np.quantile(v, 0.95):.4f}')
        print(f'  tau_crit that would discard 10% of leaves: {np.quantile(v, 0.90):.4f}')
        print(f'  tau_crit that would discard 20% of leaves: {np.quantile(v, 0.80):.4f}')


if __name__ == '__main__':
    main()
