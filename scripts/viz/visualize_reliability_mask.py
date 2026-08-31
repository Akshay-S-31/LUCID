"""Visualise the Colabator reliability mask M as a red/yellow/green overlay.

Inference only. Nothing in the existing codebase is modified -- the mask is
recomputed here exactly as corun_colabator/models/colabator_by_depth_model.py
::labal_selection builds `teacher_mask`, using the same config block from
dehazing_options/train_corun_with_colabator_by_depth.yml:

    nr_iqa   = musiq per block, nr_iqa_scale [0,100], better = higher
    clip     = daclip p(hazy) per block, degradation_type [1], better = lower
    gate     = (block-mean phys_error < uncertainty_threshold)   [ASM gate]
    M        = ((nr_iqa_mask + clip_mask) / (len(degradation_type) + 1)) * gate

Example:
    PYTHONPATH=.:basicsr_modified:corun_colabator python3 visualize_reliability_mask.py \
        --pool 24 --pick 3
"""
import argparse
import csv
import os
from glob import glob

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import pyiqa
from corun_colabator.archs.corun_arch import CORUN

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# ---------------------------------------------------------------- user's function, verbatim
def overlay_mask(hazy_img, mask, alpha=0.5):
    # mask: HxW float in [0,1]
    # colormap: 0=red, 0.5=yellow, 1=green
    heatmap = plt.cm.RdYlGn(mask)[..., :3]  # HxWx3, RGB
    overlay = alpha * heatmap + (1 - alpha) * (hazy_img / 255.0)
    return (overlay * 255).astype(np.uint8)
# ------------------------------------------------------------------------------------------


def build_args():
    p = argparse.ArgumentParser(description='Colabator reliability mask M visualiser')
    p.add_argument('--opt', default='dehazing_options/valid_corun.yml', type=str)
    p.add_argument('--train_opt', default='dehazing_options/train_corun_with_colabator_by_depth.yml', type=str)
    p.add_argument('--weights', default='experiments/trainS2_CORUN_RIDCP500/models/net_g_5000.pth', type=str)
    p.add_argument('--input_dir', default='Datasets/URHI', type=str)
    p.add_argument('--images', default=None, nargs='*')
    p.add_argument('--out_dir', default='reliability_mask_vis', type=str)
    p.add_argument('--pool', default=24, type=int, help='candidates to score before picking examples')
    p.add_argument('--pick', default=3, type=int, help='examples in the stacked figure (by low/mid/high mean M)')
    p.add_argument('--threshold', default=0.05, type=float, help='colabator uncertainty_threshold')
    p.add_argument('--alpha', default=0.5, type=float)
    p.add_argument('--max_side', default=512, type=int)
    p.add_argument('--device', default='auto', type=str)
    p.add_argument('--chunk', default=64, type=int, help='blocks per forward pass')
    return p.parse_args()


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_daclip(cfg, device):
    """Load daclip without touching the hardcoded .cuda() in the vendored open_clip."""
    from corun_colabator.archs import open_clip
    saved = nn.Module.cuda
    nn.Module.cuda = lambda self, *a, **k: self          # neutralise transformer.py:297
    try:
        model, preprocess = open_clip.create_model_from_pretrained(
            cfg['clip_model_type'], pretrained=cfg['pretrained_clip_weight'])
    finally:
        nn.Module.cuda = saved
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg['tokenizer_type'])
    degradations = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy',
                    'raindrop', 'rainy', 'shadowed', 'snowy', 'uncompleted']
    with torch.no_grad():
        tf = model.encode_text(tokenizer(degradations).to(device))
        tf /= tf.norm(dim=-1, keepdim=True)
    return model, preprocess, tf


def block_image(image, bs):
    """Same reshape as colabator_by_depth_model.block_image (B is 1 here)."""
    B, C, H, W = image.size()
    nh, nw = H // bs, W // bs
    return image.view(B, C, nh, bs, nw, bs).permute(2, 4, 0, 1, 3, 5).contiguous().view(nh * nw * B, C, bs, bs)


def unblock(seq, nh, nw, hw):
    """Same as unblock_image: per-block scalar -> bilinear upsample to full res."""
    m = seq.view(1, 1, nh, nw)
    return F.interpolate(m, hw, mode='bilinear', align_corners=False)


def chunked(fn, x, n):
    return torch.cat([fn(x[i:i + n]) for i in range(0, x.shape[0], n)], dim=0)


def main():
    args = build_args()
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'===> device: {device}')

    ccfg = yaml.load(open(args.train_opt), Loader=Loader)['colabator']
    bs = ccfg.get('block_size', 32)
    n_deg = len(ccfg['degradation_type'])
    scale_lo, scale_hi = ccfg['nr_iqa_scale']
    mode = ccfg.get('weight_map_calculation', 'addition')
    print(f"===> colabator: block={bs} nr_iqa={ccfg['nr_iqa_type']}({ccfg['nr_iqa_better']}) "
          f"clip_better={ccfg['clip_better']} combine={mode} thr={args.threshold}")

    ncfg = dict(yaml.load(open(args.opt), Loader=Loader)['network_g'])
    ncfg.pop('type')
    net = CORUN(**ncfg)
    ck = torch.load(args.weights, map_location='cpu', weights_only=True)
    net.load_state_dict(ck['params_ema' if 'params_ema' in ck else 'params'])
    net = net.to(device).eval()
    print(f'===> loaded {args.weights}')

    nr_iqa = pyiqa.create_metric(ccfg['nr_iqa_type'], device=device).eval()
    clip_model, clip_pre, text_features = load_daclip(ccfg, device)
    print('===> musiq + daclip ready')

    if args.images:
        files = args.images
    else:
        files = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            files += glob(os.path.join(args.input_dir, ext))
            files += glob(os.path.join(args.input_dir, '*', ext))
        files = sorted(files)[:args.pool]
    print(f'===> scoring {len(files)} image(s)')

    records = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        if args.max_side and max(bgr.shape[:2]) > args.max_side:
            s = args.max_side / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, (int(round(bgr.shape[1] * s)), int(round(bgr.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        x = torch.from_numpy(np.float32(bgr) / 255.).permute(2, 0, 1)[None].to(device)
        xp = F.pad(x, (0, (bs - w % bs) % bs, 0, (bs - h % bs) % bs), mode='reflect')
        H, W = xp.shape[-2:]
        nh, nw = H // bs, W // bs

        with torch.no_grad():
            outs, trans = net(xp, finetune=True)
            pseudo = outs[0].clamp(0, 1)
            t_hat = trans[0]

            # --- ASM gate (thesis contribution) ---
            phys_error = torch.mean((pseudo * t_hat + (1 - t_hat) - xp) ** 2, dim=1, keepdim=True)
            block_err = F.avg_pool2d(phys_error, bs, bs).flatten()
            gate_seq = (block_err < args.threshold).float()
            gate = unblock(gate_seq, nh, nw, (H, W))

            blocks = block_image(pseudo, bs)

            # --- NR-IQA term (musiq, better = higher) ---
            iqa_seq = chunked(lambda b: nr_iqa(b).flatten(), blocks, args.chunk)
            iqa = (unblock(iqa_seq, nh, nw, (H, W)) - scale_lo) / (scale_hi - scale_lo)
            if ccfg['nr_iqa_better'] != 'higher':
                iqa = 1 - iqa

            # --- CLIP term (daclip p(hazy), better = lower) ---
            def clip_score(b):
                _, df = clip_model.encode_image(clip_pre(b), control=True)
                df = df / df.norm(dim=-1, keepdim=True)
                probs = (100.0 * df @ text_features.T).softmax(dim=-1)
                return probs[:, ccfg['degradation_type']].sum(dim=-1)
            clip_seq = chunked(clip_score, blocks, args.chunk)
            clip_m = n_deg - unblock(clip_seq, nh, nw, (H, W))
            if ccfg['clip_better'] != 'higher':
                clip_m = n_deg - clip_m

            if mode == 'multiplication':
                M = iqa * (clip_m / n_deg) * gate
            else:
                M = ((iqa + clip_m) / (n_deg + 1)) * gate

        crop = lambda t: t[0, 0, :h, :w].float().cpu().numpy()
        rec = dict(path=path, name=name,
                   rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                   out=cv2.cvtColor((pseudo[0, :, :h, :w].float().cpu().permute(1, 2, 0).numpy() * 255)
                                    .astype(np.uint8), cv2.COLOR_BGR2RGB),
                   M=np.clip(crop(M), 0, 1).astype(np.float64),
                   iqa=crop(iqa), clip=crop(clip_m), gate=crop(gate), err=crop(phys_error))
        rec['M_mean'] = float(rec['M'].mean())
        records.append(rec)
        print(f"  {name:28s} M mean={rec['M_mean']:.3f}  musiq={rec['iqa'].mean():.3f}  "
              f"clip={rec['clip'].mean():.3f}  gate={rec['gate'].mean():.3f}")

    if not records:
        print('no readable images'); return

    # per-image component breakdown
    for r in records:
        fig, ax = plt.subplots(1, 6, figsize=(26, 4.4))
        for a in ax:
            a.axis('off')
        ax[0].imshow(r['rgb']);  ax[0].set_title('hazy input')
        ax[1].imshow(r['out']);  ax[1].set_title('teacher pseudo-label')
        for i, (k, ttl) in enumerate([('iqa', 'musiq term'), ('clip', 'daclip term'), ('gate', 'ASM gate')]):
            im = ax[2 + i].imshow(r[k], cmap='RdYlGn', vmin=0, vmax=1)
            ax[2 + i].set_title(f"{ttl}  (mean {r[k].mean():.3f})")
            fig.colorbar(im, ax=ax[2 + i], fraction=0.046)
        ax[5].imshow(overlay_mask(r['rgb'], r['M'], args.alpha))
        ax[5].set_title(f"final mask M  (mean {r['M_mean']:.3f})")
        fig.suptitle(r['name'])
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{r['name']}_components.png"), dpi=110, bbox_inches='tight')
        plt.close(fig)

    # stacked 3-panel thesis figure: low / mid / high mean M
    records.sort(key=lambda r: r['M_mean'])
    k = min(args.pick, len(records))
    idx = [int(round(i * (len(records) - 1) / max(k - 1, 1))) for i in range(k)]
    chosen = [records[i] for i in dict.fromkeys(idx)]

    fig, ax = plt.subplots(len(chosen), 3, figsize=(13, 4.1 * len(chosen)), squeeze=False)
    for row, r in enumerate(chosen):
        for c in range(3):
            ax[row][c].axis('off')
        ax[row][0].imshow(r['rgb'])
        ax[row][1].imshow(overlay_mask(r['rgb'], r['M'], args.alpha))
        ax[row][2].imshow(r['out'])
        if row == 0:
            ax[row][0].set_title('Real hazy input', fontsize=13)
            ax[row][1].set_title('Reliability mask $M$  (green = accepted)', fontsize=13)
            ax[row][2].set_title('LUCID output', fontsize=13)
        ax[row][0].set_ylabel(r['name'])
        ax[row][1].text(0.02, 0.04, f"mean $M$ = {r['M_mean']:.3f}", transform=ax[row][1].transAxes,
                        fontsize=11, color='w',
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
    fig.tight_layout()
    out_fig = os.path.join(args.out_dir, 'reliability_mask.png')
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(args.out_dir, 'reliability_mask_summary.csv'), 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['image', 'M_mean', 'musiq_mean', 'clip_mean', 'gate_mean', 'phys_err_mean'])
        for r in records:
            wtr.writerow([os.path.basename(r['path']), f"{r['M_mean']:.6f}", f"{r['iqa'].mean():.6f}",
                          f"{r['clip'].mean():.6f}", f"{r['gate'].mean():.6f}", f"{r['err'].mean():.6e}"])

    print(f"\n===> thesis figure: {out_fig}")
    print(f"===> {len(records)} component breakdowns + summary CSV in {args.out_dir}/")


if __name__ == '__main__':
    main()
