"""Driver for the LUCID ablation study.

Generates one resolved config per ablation variant, trains each in sequence,
evaluates it, and appends a row to results/experiments.csv. Nothing under
corun_colabator/ is modified: each run is defined purely by config overrides,
so the code path is identical across variants and the only thing that changes
is the hyperparameter under test.

The seven variants give two component ablations plus a three-point sweep over
each of the router's two thresholds:

    full            control -- unmodified LUCID
    no_quadtree     use_quadtree=false; falls back to the flat block gate
    no_mc_dropout   mc_K=1; teacher is deterministic, variances are zero
    tau_q_0.15      more aggressive splitting
    tau_q_0.50      less aggressive splitting
    tau_crit_0.50   discards more leaves
    tau_crit_0.90   discards fewer leaves

tau_q and tau_crit at their defaults (0.3 / 0.7) come from the `full` run, so
each sweep has three points without extra training.

Ablations are trained for fewer iterations than the published model (15k by
default rather than 40k). This is sound as long as every variant uses the same
budget, and it is well matched to this pipeline: the best-BRISQUE checkpoint of
the published run was at 5k, so the informative region is early.

Usage:
    # write configs and print the commands, without training (no GPU needed)
    python3 scripts/analysis/run_ablations.py --dry-run

    # train everything, 15k iterations each
    PYTHONPATH=.:basicsr_modified:corun_colabator \
        python3 scripts/analysis/run_ablations.py --iters 15000

    # a single variant
    PYTHONPATH=.:basicsr_modified:corun_colabator \
        python3 scripts/analysis/run_ablations.py --only no_quadtree
"""

import argparse
import copy
import glob
import os
import re
import shutil
import subprocess
import sys

import yaml

BASE_OPT = 'dehazing_options/train_corun_with_colabator_by_depth.yml'
CONFIG_DIR = 'experiments/ablation_configs'
CSV = 'results/experiments.csv'

# (variant name, {dotted config path: value})
VARIANTS = [
    ('full',            {}),
    ('no_quadtree',     {'colabator.use_quadtree': False}),
    ('no_mc_dropout',   {'colabator.mc_K': 1}),
    ('tau_q_0.15',      {'colabator.tau_q': 0.15}),
    ('tau_q_0.50',      {'colabator.tau_q': 0.50}),
    ('tau_crit_0.50',   {'colabator.tau_crit': 0.50}),
    ('tau_crit_0.90',   {'colabator.tau_crit': 0.90}),
]

# Applied to every variant. Validation image writing is the one setting that can
# quietly fill a disk: at val_freq 5000 with save_img on, each run dumps the
# whole RTTS set to disk several times over.
COMMON = {
    'val.save_img': False,
    'val.save_source': False,
    'logger.save_checkpoint_freq': 5000,
}


def set_path(d, dotted, value):
    """Set d['a']['b'] = value for dotted == 'a.b', erroring on unknown keys."""
    keys = dotted.split('.')
    node = d
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f'{dotted}: no such section {k!r} in the base config')
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f'{dotted}: no such key {keys[-1]!r} in the base config')
    node[keys[-1]] = value


def build_config(name, overrides, iters):
    """Write a resolved config for one variant and return its path."""
    with open(BASE_OPT) as f:
        opt = yaml.safe_load(f)

    opt = copy.deepcopy(opt)
    opt['name'] = f'ablation_{name}'

    # Keep the three iteration counts consistent. total_iter drives the loop,
    # datasets.train.iters drives the progressive-batch schedule, and the
    # scheduler period must match or the learning rate will not complete its
    # cosine cycle within the run.
    set_path(opt, 'train.total_iter', iters)
    set_path(opt, 'datasets.train.iters', [iters])
    set_path(opt, 'train.gen_scheduler.periods', [iters])

    for dotted, value in COMMON.items():
        set_path(opt, dotted, value)
    for dotted, value in overrides.items():
        set_path(opt, dotted, value)

    os.makedirs(CONFIG_DIR, exist_ok=True)
    path = os.path.join(CONFIG_DIR, f'{name}.yml')
    with open(path, 'w') as f:
        yaml.safe_dump(opt, f, sort_keys=False, default_flow_style=False)
    return path


def newest_checkpoint(run_name):
    """Highest-iteration net_g_*.pth for a finished run, or None."""
    pat = f'experiments/ablation_{run_name}/models/net_g_*.pth'
    best, best_it = None, -1
    for p in glob.glob(pat):
        m = re.search(r'net_g_(\d+)\.pth$', p)
        if m and int(m.group(1)) > best_it:
            best, best_it = p, int(m.group(1))
    return best


def retention_curve(run_name):
    """Mean data_retention over the last 20 logged points of a run."""
    logs = sorted(glob.glob(f'experiments/ablation_{run_name}/*.log'))
    if not logs:
        return None
    vals = []
    with open(logs[-1]) as f:
        for line in f:
            m = re.search(r'data_retention: ([0-9.eE+-]+)', line)
            if m:
                vals.append(float(m.group(1)))
    if not vals:
        return None
    tail = vals[-20:]
    return sum(tail) / len(tail)


def prune(run_name):
    """Drop optimizer states and intermediate checkpoints; keep the last one."""
    root = f'experiments/ablation_{run_name}'
    states = os.path.join(root, 'training_states')
    if os.path.isdir(states):
        shutil.rmtree(states)
    keep = newest_checkpoint(run_name)
    for p in glob.glob(os.path.join(root, 'models', 'net_g_*.pth')):
        if p != keep:
            os.remove(p)


def main():
    p = argparse.ArgumentParser(description='LUCID ablation driver')
    p.add_argument('--iters', type=int, default=15000,
                   help='iterations per ablation run (default 15000)')
    p.add_argument('--only', default=None, help='run a single variant by name')
    p.add_argument('--dry-run', action='store_true',
                   help='write configs and print commands without training')
    p.add_argument('--no-prune', action='store_true',
                   help='keep optimizer states and all checkpoints')
    p.add_argument('--no-eval', action='store_true',
                   help='skip the evaluation step after each run')
    args = p.parse_args()

    variants = VARIANTS
    if args.only:
        variants = [v for v in VARIANTS if v[0] == args.only]
        if not variants:
            sys.exit(f'unknown variant {args.only!r}; '
                     f'choose from {[v[0] for v in VARIANTS]}')

    est_h = args.iters * 1.1 / 3600.0
    print(f'{len(variants)} run(s), {args.iters} iterations each')
    print(f'rough estimate: {est_h:.1f} h per run, '
          f'{est_h * len(variants):.1f} h total on a single 4090\n')

    for name, overrides in variants:
        cfg = build_config(name, overrides, args.iters)
        desc = ', '.join(f'{k}={v}' for k, v in overrides.items()) or 'control'
        print(f'=== {name}  ({desc})')
        print(f'    config: {cfg}')

        cmd = [sys.executable, 'corun_colabator/train.py', '-opt', cfg]
        if args.dry_run:
            print(f'    would run: {" ".join(cmd)}\n')
            continue

        env = dict(os.environ, HF_HUB_OFFLINE='True')
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            print(f'    TRAINING FAILED (exit {rc}) -- continuing to next variant\n')
            continue

        ret = retention_curve(name)
        if ret is not None:
            print(f'    mean data_retention (last 20 logs): {ret:.4f}')

        ckpt = newest_checkpoint(name)
        if ckpt and not args.no_eval:
            print(f'    evaluating {ckpt}')
            subprocess.call([
                sys.executable, 'scripts/eval/lucid_stream_eval.py',
                '--weights', ckpt,
                '--experiment_id', f'ablation_{name}',
                '--variant', desc,
                '--best_checkpoint_iter', re.search(r'(\d+)\.pth$', ckpt).group(1),
                '--data_retention_mean', f'{ret:.4f}' if ret is not None else '',
                '--datasets', 'sots_indoor,rtts',
            ], env=env)

        if not args.no_prune:
            prune(name)
        print()

    print(f'done. numeric results appended to {CSV}')
    print('retention curves can be replotted from experiments/ablation_*/*.log')


if __name__ == '__main__':
    main()
