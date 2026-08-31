"""Visualise the ASM gate (physical drift error) as a red/yellow/green reliability
overlay. Inference only -- no training, no changes to any existing module.

The reliability signal is the same `phys_error` computed in
corun_colabator/models/colabator_by_depth_model.py::optimize_parameters:

    I_recon    = pseudo_label * pseudo_transmission + (1 - pseudo_transmission)
    phys_error = mean((I_recon - hazy) ** 2, dim=1, keepdim=True)

and the block gate is the same `confidence_gate_mask` built in labal_selection
(block mean -> threshold -> bilinear upsample).

Example:
    PYTHONPATH=.:basicsr_modified python3 visualize_asm_gate.py \
        --weights experiments/trainS2_CORUN_RIDCP500/models/net_g_5000.pth \
        --num 8
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
import torch.nn.functional as F
import yaml

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
    p = argparse.ArgumentParser(description='ASM gate / reliability map visualiser')
    p.add_argument('--opt', default='dehazing_options/valid_corun.yml', type=str)
    p.add_argument('--weights', default='experiments/trainS2_CORUN_RIDCP500/models/net_g_5000.pth', type=str)
    p.add_argument('--input_dir', default='Datasets/RTTS/RTTS/JPEGImages', type=str)
    p.add_argument('--images', default=None, nargs='*', help='explicit image paths (overrides --input_dir)')
    p.add_argument('--out_dir', default='asm_gate_vis', type=str)
    p.add_argument('--num', default=8, type=int, help='how many images from --input_dir')
    p.add_argument('--block_size', default=32, type=int, help='colabator block_size')
    p.add_argument('--threshold', default=0.05, type=float, help='colabator uncertainty_threshold')
    p.add_argument('--alpha', default=0.5, type=float)
    p.add_argument('--vmax', default='auto', type=str,
                   help="error value mapped to pure red: 'auto' (99th pct per image), 'gate' (=threshold), or a float")
    p.add_argument('--device', default='auto', type=str, help='auto | mps | cpu | cuda')
    p.add_argument('--max_side', default=640, type=int, help='downscale long side before inference (0 = native)')
    p.add_argument('--stacked', action='store_true',
                   help='also emit the stacked 3-panel thesis figure (hazy | gate overlay | output)')
    p.add_argument('--pick', default=3, type=int, help='examples in the stacked figure, spread by retention')
    return p.parse_args()


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(opt_path, weights, device):
    cfg = yaml.load(open(opt_path, mode='r'), Loader=Loader)
    net_cfg = dict(cfg['network_g'])
    net_cfg.pop('type')
    model = CORUN(**net_cfg)
    ckpt = torch.load(weights, map_location='cpu', weights_only=True)
    key = 'params_ema' if 'params_ema' in ckpt else 'params'
    model.load_state_dict(ckpt[key])
    print(f'===> loaded {weights} [{key}]')
    return model.to(device).eval()


def main():
    args = build_args()
    device = pick_device(args.device)
    print(f'===> device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.opt, args.weights, device)

    if args.images:
        files = args.images
    else:
        files = sorted(glob(os.path.join(args.input_dir, '*.png')) +
                       glob(os.path.join(args.input_dir, '*.jpg')))[:args.num]
    print(f'===> {len(files)} image(s)')

    bs = args.block_size
    rows = []
    panels = []

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            print(f'  !! unreadable: {path}')
            continue

        if args.max_side and max(bgr.shape[:2]) > args.max_side:
            s = args.max_side / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, (int(round(bgr.shape[1] * s)), int(round(bgr.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)

        h, w = bgr.shape[:2]
        x = torch.from_numpy(np.float32(bgr) / 255.).permute(2, 0, 1).unsqueeze(0).to(device)

        # pad to a multiple of block_size so block_image's view() is exact
        ph = (bs - h % bs) % bs
        pw = (bs - w % bs) % bs
        xp = F.pad(x, (0, pw, 0, ph), mode='reflect')

        with torch.no_grad():
            outs, trans = model(xp, finetune=True)
            dehazed = outs[0].clamp(0, 1)
            t_hat = trans[0]
            # ASM gate, exactly as in optimize_parameters
            i_recon = dehazed * t_hat + (1 - t_hat)
            phys_error = torch.mean((i_recon - xp) ** 2, dim=1, keepdim=True)

            # block gate, exactly as in labal_selection
            block_err = F.avg_pool2d(phys_error, bs, bs)
            gate = (block_err < args.threshold).float()
            gate_mask = F.interpolate(gate, phys_error.shape[-2:], mode='bilinear', align_corners=False)

        err = phys_error[0, 0, :h, :w].float().cpu().numpy()
        gate_np = gate_mask[0, 0, :h, :w].float().cpu().numpy()
        t_np = t_hat[0, 0, :h, :w].float().cpu().numpy()
        deh_bgr = (dehazed[0, :, :h, :w].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        retention = float(gate.mean().item())

        if args.vmax == 'auto':
            vmax = float(np.percentile(err, 99)) or 1e-8
        elif args.vmax == 'gate':
            vmax = args.threshold
        else:
            vmax = float(args.vmax)

        # reliability in [0,1]: 1 = physically consistent (green), 0 = drift (red)
        reliability = np.clip(1.0 - err / vmax, 0.0, 1.0).astype(np.float64)

        # cv2 gives BGR; overlay_mask blends against an RGB colormap
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        deh_rgb = cv2.cvtColor(deh_bgr, cv2.COLOR_BGR2RGB)

        ov_cont = overlay_mask(rgb, reliability, args.alpha)
        ov_gate = overlay_mask(rgb, gate_np.astype(np.float64), args.alpha)

        cv2.imwrite(os.path.join(args.out_dir, f'{name}_overlay.png'),
                    cv2.cvtColor(ov_cont, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out_dir, f'{name}_gate.png'),
                    cv2.cvtColor(ov_gate, cv2.COLOR_RGB2BGR))

        fig, ax = plt.subplots(1, 5, figsize=(22, 4.4))
        for a in ax:
            a.axis('off')
        ax[0].imshow(rgb);      ax[0].set_title('hazy input')
        ax[1].imshow(deh_rgb);  ax[1].set_title('LUCID output')
        im = ax[2].imshow(t_np, cmap='magma', vmin=0, vmax=1)
        ax[2].set_title(r'transmission $\hat{t}$')
        fig.colorbar(im, ax=ax[2], fraction=0.046)
        ax[3].imshow(ov_cont)
        ax[3].set_title(f'reliability (vmax={vmax:.2e})\ngreen = ASM-consistent')
        ax[4].imshow(ov_gate)
        ax[4].set_title(f'block gate @ thr={args.threshold}\nretention {retention*100:.1f}%')
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'{name}_panel.png'), dpi=110, bbox_inches='tight')
        plt.close(fig)

        panels.append(dict(name=name, rgb=rgb, out=deh_rgb, ov_gate=ov_gate, retention=retention))
        rows.append(dict(image=os.path.basename(path), h=h, w=w,
                         err_mean=err.mean(), err_p99=np.percentile(err, 99), err_max=err.max(),
                         retention_rate=retention, vmax=vmax))
        print(f'  {name:24s} err mean={err.mean():.3e} p99={np.percentile(err,99):.3e} '
              f'max={err.max():.3e}  retention={retention*100:.1f}%')

    if args.stacked and panels:
        panels.sort(key=lambda r: r['retention'])
        k = min(args.pick, len(panels))
        idx = [int(round(i * (len(panels) - 1) / max(k - 1, 1))) for i in range(k)]
        chosen = [panels[i] for i in dict.fromkeys(idx)]

        fig, ax = plt.subplots(len(chosen), 3, figsize=(13, 4.1 * len(chosen)), squeeze=False)
        for row, r in enumerate(chosen):
            for c in range(3):
                ax[row][c].axis('off')
            ax[row][0].imshow(r['rgb'])
            ax[row][1].imshow(r['ov_gate'])
            ax[row][2].imshow(r['out'])
            if row == 0:
                ax[row][0].set_title('Real hazy input', fontsize=13)
                ax[row][1].set_title('ASM gate  (green = pseudo-label accepted)', fontsize=13)
                ax[row][2].set_title('LUCID output', fontsize=13)
            ax[row][1].text(0.02, 0.04, f"retention = {r['retention']*100:.1f}%",
                            transform=ax[row][1].transAxes, fontsize=11, color='w',
                            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
        fig.tight_layout()
        out_fig = os.path.join(args.out_dir, 'asm_gate_figure.png')
        fig.savefig(out_fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'\n===> stacked thesis figure: {out_fig}')

    if rows:
        with open(os.path.join(args.out_dir, 'asm_gate_summary.csv'), 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f'\n===> wrote {len(rows)*3 + 1} files to {args.out_dir}/')
        print(f'===> mean retention across images: '
              f'{100*np.mean([r["retention_rate"] for r in rows]):.1f}%')


if __name__ == '__main__':
    main()
