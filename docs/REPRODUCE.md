# Reproducing the LUCID results

Every command below is run from the repository root with the environment from
the README already installed.

```bash
export PYTHONPATH=.:basicsr_modified
```

The released LUCID checkpoint is the best-BRISQUE checkpoint at 5,000
fine-tuning iterations, distributed as `LUCID.pth` from the Google Drive link in
the README. Place it at `./pretrained_weights/LUCID.pth`. The CORUN+ baseline
checkpoint is `./CORUN+.pth`.

---

## A note on evaluation resolution

Paired metrics depend on the grid the two images are compared on, so the
resolution is an explicit flag rather than a hidden default.

- `--size 0` (default) compares at the **ground truth's own resolution**,
  resampling the result to match only if the two differ.
- `--size 256` puts **both** images on a 256×256 grid.

The I-Haze and O-Haze tables use `--size 256`, matching the resolution the
results in `results/` were generated at. Using the wrong flag changes
PSNR/SSIM/LPIPS substantially, so it is stated for each table below.

No-reference metrics (BRISQUE, NIMA, NIQE, MUSIQ) are computed on the result as
produced and are unaffected by this flag.

---

## 1. Inference

Generate dehazed outputs for a dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python3 corun_colabator/simple_test.py \
  --opt dehazing_options/valid_corun.yml \
  --input_dir Datasets/RTTS/RTTS/JPEGImages \
  --result_dir ./results/LUCID_RTTS \
  --weights ./pretrained_weights/LUCID.pth \
  --dataset RTTS
```

`simple_test.py` runs at native resolution by default. Add `--size 256` only if
you need a fixed smaller input to fit in memory; it changes the output
resolution and therefore the metrics.

---

## 2. RTTS — unpaired evaluation

FADE, BRISQUE and NIMA over all 4,322 RTTS images.

```bash
# BRISQUE / NIMA / NIQE / MUSIQ on the generated outputs
python3 scripts/eval/evaluate.py --input_dir ./results/LUCID_RTTS

# FADE
python3 scripts/eval/../../fade_results/run_fade_rtts.py
python3 fade_results/fade_stats.py
```

`fade_results/fade_summary.json` in this repository holds the FADE run over the
full 4,322-image set for the hazy input, the CORUN+ baseline and both LUCID
checkpoints.

---

## 3. SOTS-Indoor — paired evaluation

`Datasets/SOTS/indoor/gt_aligned/` holds one ground-truth copy per hazy image,
so filenames pair one-to-one; `gt/` holds the 50 unique scenes.

```bash
python3 scripts/eval/evaluate_full.py \
  --input_dir ./results/LUCID_SOTS \
  --gt_dir Datasets/SOTS/indoor/gt_aligned
```

Or stream the evaluation without writing any images to disk:

```bash
python3 scripts/eval/lucid_stream_eval.py \
  --weights ./pretrained_weights/LUCID.pth \
  --experiment_id lucid --variant released_5k \
  --datasets sots_indoor
```

---

## 4. I-Haze and O-Haze — paired evaluation

Uses `--size 256`. Both stages are wrapped in one script:

```bash
sh scripts/run_evaluations.sh
```

Results are appended to `eval_results.txt`. To run a single set directly:

```bash
python3 scripts/eval/evaluate_full.py \
  --input_dir results/ihaze_lucid/ihaze \
  --gt_dir Datasets/ihaze/I-HAZE/test/clear \
  --size 256
```

This reproduces the I-Haze table exactly:

| | PSNR | SSIM | LPIPS | BRISQUE | NIMA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CORUN+ | 17.2666 | 0.8127 | 0.1765 | 4.8449 | 4.5136 |
| LUCID | 17.4835 | 0.8094 | 0.1763 | 4.5264 | 4.3745 |

---

## 5. Extended no-reference metrics

NIQE and MUSIQ are reported alongside every paired run by
`scripts/eval/evaluate_full.py`. No separate command is needed.

---

## 6. Computational cost

```bash
python3 scripts/eval/benchmark_efficiency.py \
  --weights CORUN+.pth ./pretrained_weights/LUCID.pth
```

Reports parameter count and 1080p throughput for each checkpoint. Both report
18.6743 M parameters and the same throughput, since LUCID's contribution is
training-time only and MC Dropout is inert under `eval()`. Absolute FPS depends
on the device; the paper's figure is measured on an RTX 4090.

---

## 7. ASM gate retention sweep

Measures the fraction of blocks passing the gate on URHI as a function of
`tau_ASM`, at the pretrained teacher:

```bash
python3 scripts/analysis/lucid_retention_sweep.py \
  --taus 0.02,0.05,0.10,0.15,0.20 --limit 500
```

Appends to `results/experiments.csv`.

---

## 8. Reliability visualisations

```bash
# ASM gate / physical drift overlay
python3 scripts/viz/visualize_asm_gate.py \
  --weights ./pretrained_weights/LUCID.pth --num 8

# Full reliability mask M
python3 scripts/viz/visualize_reliability_mask.py --pool 24 --pick 3
```

Both write panels plus a summary CSV into `asm_gate_vis/` and
`reliability_mask_vis/`. These directories are generated output and are not
tracked in git.

---

## 9. Training from scratch

```bash
# Stage 1 — synthetic pre-training on RIDCP500
sh dehazing_options/train_corun_by_depth_single_gpu.sh

# Stage 2 — semi-supervised fine-tuning on URHI
sh dehazing_options/train_corun_with_colabator_by_depth_single_gpu.sh
```

Set `path.pretrain_network_g` in the stage-2 config to the stage-1 checkpoint
before starting. All LUCID hyperparameters — `mc_K`, `uncertainty_threshold`
(`tau_ASM`), `use_quadtree`, `quadtree_sizes`, `tau_q`, `tau_crit` and
`network_g.mc_dropout_p` — are set in
`dehazing_options/train_corun_with_colabator_by_depth.yml`, annotated with the
paper section each corresponds to. See [METHOD.md](METHOD.md) for the mapping
from each equation to its implementation.
