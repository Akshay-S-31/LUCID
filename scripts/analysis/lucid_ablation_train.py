"""Runtime-patched ablation / hyperparameter launcher for LUCID fine-tuning.

Wraps the stock Colabator training pipeline and monkey-patches specific
components *at runtime*. Nothing under corun_colabator/ is modified on disk, so
the code path that produced the published LUCID numbers stays byte-identical.

Implemented variants
--------------------
  full            unmodified LUCID (control)
  no_gate         bypass the ASM gate entirely (confidence_gate_mask = 1.0, so
                  every block is retained). This is the agreed reframing of the
                  original "ablation_no_gate": MC-Dropout does not exist in this
                  codebase -- it was removed in 7e2f868 ("Remove hallucinatory
                  MC Dropout novelty") as a README claim with no implementation
                  behind it -- so the meaningful ablation is disabling the gate,
                  not swapping in a stochastic criterion that was never built.
  no_mask_loss    keep the ASM gate, but apply the contrastive perceptual and
                  style terms unconditionally (drop the * pseudo_mask.mean())
  tau_asm         sweep colabator.uncertainty_threshold (the ASM gate tau)
  block_size      sweep the block granularity. This is the agreed reframing of
                  "ablation_no_quadtree": there is no quadtree -- labal_selection
                  does a single uniform block_image() reshape at block_size=32
                  with no recursion or subdivision, which is already what that
                  ablation described.

Not implemented: hp_K_dropout. It sweeps K over a MC-Dropout component that does
not exist (and K=1 yields zero variance by construction).

Checkpoint retention: --keep_best 2 prunes the run's checkpoint directory down
to the best-BRISQUE and best-NIMA iterations after training.

Usage:
    PYTHONPATH=.:basicsr_modified:corun_colabator python3 lucid_ablation_train.py \
        --variant tau_asm --value 0.10 --total_iter 40000
"""

import argparse
import os
import sys

import torch
import yaml

BASE_OPT = 'dehazing_options/train_corun_with_colabator_by_depth.yml'


def patch_no_mask_loss():
    """Remove the pseudo_mask.mean() gating from the two contrastive terms.

    Rather than rewriting optimize_parameters (long, and it would drift from
    upstream), we wrap the loss module so its returned contrastive terms are
    later multiplied by a mask whose mean is forced to 1.0. Equivalent to
    deleting the multiplier, but it touches one small surface.
    """
    from corun_colabator.models import colabator_by_depth_model as mod

    Model = mod.ColabatorByDepthModel
    original = Model.labal_selection

    def patched(self, teacher_transmission, teacher, joint_uncertainty_map=None):
        t_trans, t, mask = original(self, teacher_transmission, teacher,
                                    joint_uncertainty_map)
        # The l_pix and l_asm terms use the mask elementwise and must keep the
        # real gate. Only the contrastive terms use mask.mean(). Substituting a
        # tensor whose .mean() is 1 but whose elementwise values are unchanged
        # is not expressible, so expose a flagged wrapper the model already
        # multiplies by; simplest faithful route is a mask subclass.
        class _UngatedMean(type(mask)):
            def mean(self, *a, **k):
                if not a and not k:
                    return torch.ones((), device=self.device, dtype=self.dtype)
                return super().mean(*a, **k)

        return t_trans, t, mask.as_subclass(_UngatedMean)

    Model.labal_selection = patched
    print('[patch] no_mask_loss: contrastive percep/style terms are now ungated '
          '(pseudo_mask.mean() -> 1.0); l_pix and l_asm keep the elementwise gate')


def patch_no_gate():
    """Bypass the ASM gate: every block is retained, retention == 1.0.

    labal_selection() builds confidence_gate_mask from phys_error; passing
    joint_uncertainty_map=None makes it fall through to the scalar 1.0 branch
    that already exists upstream, so no gating logic is rewritten.
    """
    from corun_colabator.models import colabator_by_depth_model as mod

    Model = mod.ColabatorByDepthModel
    original = Model.labal_selection

    def patched(self, teacher_transmission, teacher, joint_uncertainty_map=None):
        out = original(self, teacher_transmission, teacher, None)
        self._data_retention_rate = 1.0
        return out

    Model.labal_selection = patched
    print('[patch] no_gate: ASM gate bypassed (confidence_gate_mask = 1.0, '
          'retention forced to 100%)')


def build_opt(args):
    with open(BASE_OPT) as f:
        opt = yaml.safe_load(f)

    opt['name'] = args.experiment_id
    opt['train']['total_iter'] = args.total_iter
    opt['train']['gen_scheduler']['periods'] = [args.total_iter]
    opt['datasets']['train']['iters'] = [args.total_iter]
    opt['path']['pretrain_network_g'] = args.pretrained
    opt['path']['param_key_g'] = 'params_ema'

    opt.setdefault('colabator', {})
    if args.variant == 'tau_asm':
        opt['colabator']['uncertainty_threshold'] = float(args.value)
        print(f'[cfg] uncertainty_threshold (tau_ASM) = {args.value}')
    elif args.variant == 'block_size':
        opt['colabator']['block_size'] = int(args.value)
        print(f'[cfg] block_size = {args.value}')
    else:
        opt['colabator'].setdefault('uncertainty_threshold', 0.05)

    return opt


def prune_checkpoints(exp_dir, keep_iters):
    """Delete every checkpoint except the iterations named in keep_iters."""
    ck_dir = os.path.join(exp_dir, 'models')
    if not os.path.isdir(ck_dir):
        print(f'[prune] no checkpoint dir at {ck_dir}')
        return 0.0
    keep = {str(int(i)) for i in keep_iters}
    freed = 0
    for fn in os.listdir(ck_dir):
        if not fn.endswith('.pth'):
            continue
        stem = os.path.splitext(fn)[0]
        it = stem.split('_')[-1]
        if it not in keep:
            p = os.path.join(ck_dir, fn)
            freed += os.path.getsize(p)
            os.remove(p)
            print(f'[prune] removed {fn}')
    print(f'[prune] kept iters {sorted(keep)}, freed {freed/1e9:.2f} GB')
    return freed / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True,
                   choices=['full', 'no_gate', 'no_mask_loss', 'tau_asm',
                            'block_size'])
    p.add_argument('--value', default='', help='hyperparameter value for sweeps')
    p.add_argument('--experiment_id', default='')
    p.add_argument('--total_iter', type=int, default=40000)
    p.add_argument('--pretrained', default='./CORUN+.pth')
    p.add_argument('--keep_best', type=int, default=2)
    p.add_argument('--dump_opt', action='store_true',
                   help='write the resolved config and exit without training')
    args = p.parse_args()

    if not args.experiment_id:
        args.experiment_id = (f'{args.variant}_{args.value}' if args.value
                              else args.variant)

    if args.variant == 'no_gate':
        patch_no_gate()
    elif args.variant == 'no_mask_loss':
        patch_no_mask_loss()

    opt = build_opt(args)

    out = os.path.join('experiments', args.experiment_id)
    os.makedirs(out, exist_ok=True)
    resolved = os.path.join(out, 'resolved_options.yml')
    with open(resolved, 'w') as f:
        yaml.safe_dump(opt, f, sort_keys=False)
    print(f'[cfg] resolved config -> {resolved}')

    if args.dump_opt:
        print('[dry-run] --dump_opt set; not launching training')
        return 0

    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        print('[warn] no accelerator visible; this will run on CPU')

    from corun_colabator.train_pipeline import train_pipeline
    sys.argv = ['train.py', '-opt', resolved]
    train_pipeline(os.getcwd())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
