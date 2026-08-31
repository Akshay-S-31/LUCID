# LUCID — method to code map

This document maps each component of the LUCID methodology onto the file and
function that implements it, so a reader of the paper can find the
corresponding code without searching.

LUCID does not modify the CORUN generator. Every contribution lives in the
semi-supervised fine-tuning loop, and the deployed network is architecturally
identical to CORUN+.

---

## 1. Preliminaries — the CORUN backbone

| Paper | Code |
| :--- | :--- |
| ASM in matrix form, `P = J·T + I − T` | `corun_colabator/archs/corun_arch.py` — `ProximalDehazingBlock`, `ProximalTransmissionBlock` |
| TGDM / SGDM + T-CPMM / S-CPMM, unrolled `K = 4` stages | `corun_arch.py` — `Basic_block_fix_Plus`, assembled in `CORUN` |
| End-to-end map `f_theta` | `CORUN.forward` |

`CORUN(depth=4)` builds `depth − 2 = 2` intermediate blocks plus a `first` and a
`last` block, i.e. four `Basic_block_fix_Plus` stages in total, which is where
the four MC-Dropout layers appear.

---

## 2. Teacher-only MC Dropout

| Paper | Code |
| :--- | :--- |
| Bernoulli dropout, `p = 0.1`, teacher branch only | `corun_arch.py` — `Basic_block_fix_Plus.mc_dropout`, applied in `forward` |
| Enable dropout without leaving eval mode | `corun_arch.py` — `enable_mc_dropout()` / `disable_mc_dropout()` |
| `K = 5` stochastic passes | `models/colabator_by_depth_model.py` — `mc_teacher_forward()` |
| Eq. 6 — means `J_mu`, `T_mu` | `mc_teacher_forward()`, `stack.mean(0)` |
| Eq. 7 — variances `sigma^2_J`, `sigma^2_T` | `mc_teacher_forward()`, `stack.var(0, unbiased=False)` |

`nn.Dropout` is inert under `eval()`, so the student and the deployed model are
deterministic and inference cost is unchanged. `enable_mc_dropout` switches only
the dropout layers into training mode, leaving every other layer untouched, and
`mc_teacher_forward` restores determinism in a `finally` block so a raised
forward pass cannot leave the teacher stochastic.

Setting `mc_K: 1` collapses this to a single deterministic pass with zero
variance.

---

## 3. Physical ASM gate

| Paper | Code |
| :--- | :--- |
| Eq. 4 — `I_recon = J_mu · T_mu + (1 − T_mu)` | `colabator_by_depth_model.py` — `optimize_parameters()` |
| Eq. 5 — `eps_phys = ||I_real − I_recon||²` | `optimize_parameters()`, channel-mean squared error |
| Gate `eps_phys < tau_ASM` | `labal_selection()`, the `use_quadtree: false` branch |
| `tau_ASM = 0.05` | `dehazing_options/train_corun_with_colabator_by_depth.yml` → `colabator.uncertainty_threshold` |

The drift error is computed from the teacher's own prediction, so it needs one
deterministic reconstruction rather than any extra network evaluation.

---

## 4. Joint quadtree spatial routing

| Paper | Code |
| :--- | :--- |
| Eq. 8 — `U_joint = (sigma^2_J + sigma^2_T + eps_phys) / 3` | `archs/quadtree_router.py` — `joint_uncertainty()` |
| Per-map min-max normalisation | `quadtree_router.py` — `minmax_normalise()` |
| Recursive split `64 → 32 → 16 → 8` on `tau_Q` | `quadtree_router.py` — `QuadtreeRouter.forward()` |
| Leaf discarded when above `tau_crit` | `QuadtreeRouter._paint()` |
| Per-leaf DA-CLIP + MUSIQ scoring | `colabator_by_depth_model.py` — `score_blocks()` |
| Output mask `M` | return value of `QuadtreeRouter.forward()`, consumed in `labal_selection()` |

Normalisation is per sample, so one pathological image cannot rescale the
uncertainties of its batch-mates, and a constant map degrades to "uniformly
certain" instead of producing NaN.

Leaves are grouped by block size and scored one group at a time, so a region
that stayed at 64×64 is judged at 64×64 and one split down to 8×8 is judged at
8×8. This is what the paper means by scoring children at higher spatial
resolution; resampling every leaf to a common size would discard the detail the
evaluators read.

The root grid covers the whole image including partial trailing blocks, so
inputs whose dimensions are not multiples of 64 are still fully covered.

### Configuration

```yaml
colabator:
  use_quadtree: true
  quadtree_sizes: [64, 32, 16, 8]
  tau_q: 0.3
  tau_crit: 0.7
```

`tau_q` and `tau_crit` are not given numerically in the paper. The values above
are the documented defaults, applied to `U_joint` after normalisation to `[0,1]`.
Setting `use_quadtree: false` falls back to the flat ASM gate on the uniform
`block_size` grid.

---

## 5. Mask-gated semi-supervised losses

| Paper | Code |
| :--- | :--- |
| `L_fine`, real-domain terms scaled by `M` | `colabator_by_depth_model.py` — `optimize_parameters()` |
| Pixel term gated by the mask | `l_pix += (cri_pix(real_output, pseudo_label) * pseudo_mask * 2).mean()` |
| ASM coherence term gated by the mask | `l_asm = (cri_pix(recon_real_lq, real_strong) * pseudo_mask).mean() * 0.01` |
| Contrastive perceptual and style gated by `M_bar` | `l_contrast_percep * pseudo_mask.mean()`, likewise for style |
| Synthetic-pair terms left unmasked | `l_pix = cri_pix(output, gt).mean()`, `l_percep`, `l_style` |

Gating the perceptual and style terms is what stops a rejected pseudo-label from
still pulling the student toward it in VGG feature space.

---

## 6. Training protocol

| Paper | Code |
| :--- | :--- |
| Stage 1 — RIDCP500 pre-training, 30k iterations, lr 2e-4 → 1e-6 | `dehazing_options/train_corun_by_depth.yml` |
| Stage 2 — semi-supervised fine-tuning, 40k iterations, lr 5e-5 | `dehazing_options/train_corun_with_colabator_by_depth.yml` |
| Crop size 192×192 | `datasets.train.gt_size` in both configs |
| Strong augmentation on the student's real branch | `corun_colabator/data/haze_online_dataset.py` |
| EMA teacher update | `models/sr_model.py` — `model_ema()`, `train.ema_decay` |
| Optimal label pool | `archs/memory_bank.py`, called from `labal_selection()` |

---

## 7. Reliability signals in the logs

`data_retention` is written to TensorBoard each iteration. With the quadtree
enabled it is the fraction of the image the router did not hard-zero — the area
still able to contribute gradient. With the flat gate it is the fraction of
blocks passing `tau_ASM`.
