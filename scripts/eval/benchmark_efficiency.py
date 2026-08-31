"""Reproduce the computational-cost table (paper Table "Computational cost").

Reports the generator's parameter count and its inference throughput at 1080p.
Because LUCID's contribution is entirely training-time -- the architecture is
unchanged and MC Dropout is inert under eval() -- a LUCID checkpoint and the
CORUN+ baseline are expected to produce identical numbers here. Passing both
with --weights makes that explicit rather than asserting it.

Usage:
    PYTHONPATH=.:basicsr_modified python3 scripts/eval/benchmark_efficiency.py \
        --weights CORUN+.pth experiments/trainS2_CORUN_RIDCP500/models/net_g_5000.pth
"""

import argparse
import os
import sys
import time
import types

import torch
import torch.nn.functional as F

os.environ.setdefault('HF_HUB_OFFLINE', 'True')


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
    if weights:
        ckpt = torch.load(weights, map_location='cpu')
        key = 'params_ema' if 'params_ema' in ckpt else ('params' if 'params' in ckpt else None)
        net.load_state_dict(ckpt[key] if key else ckpt, strict=True)
    net.to(device).eval()
    return net


@torch.no_grad()
def throughput(net, device, h=1080, w=1920, warmup=2, runs=5):
    """Median FPS over `runs` timed forward passes at the given resolution."""
    x = torch.rand(1, 3, h, w, device=device)
    H = ((h + 7) // 8) * 8
    W = ((w + 7) // 8) * 8
    x = F.pad(x, (0, W - w, 0, H - h), mode='reflect')

    def _sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    for _ in range(warmup):
        net(x, finetune=True)
    _sync()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        net(x, finetune=True)
        _sync()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    return 1.0 / median, median


def main():
    p = argparse.ArgumentParser(description='LUCID / CORUN+ efficiency benchmark')
    p.add_argument('--weights', nargs='*', default=['CORUN+.pth'],
                   help='One or more checkpoints. Pass none to measure the '
                        'untrained architecture.')
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--device', default=None,
                   help='cuda / mps / cpu. Defaults to the best available. '
                        'The paper reports an RTX 4090; other devices will '
                        'differ in FPS but not in parameter count.')
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_available() else 'cpu')

    print(f'device: {device}   resolution: {args.height}x{args.width}   runs: {args.runs}')
    print(f'{"checkpoint":52s} {"params (M)":>11s} {"FPS":>8s} {"s/img":>8s}')

    targets = args.weights if args.weights else [None]
    for w in targets:
        net = load_corun(w, device, depth=args.depth)
        params = sum(p_.numel() for p_ in net.parameters())
        fps, secs = throughput(net, device, args.height, args.width, runs=args.runs)
        label = os.path.basename(w) if w else '<untrained architecture>'
        print(f'{label:52s} {params / 1e6:11.4f} {fps:8.2f} {secs:8.3f}')
        del net


if __name__ == '__main__':
    main()
