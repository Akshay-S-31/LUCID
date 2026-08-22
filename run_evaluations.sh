#!/bin/bash
set -e

echo "Starting Inference on I-Haze..."
PYTHONPATH=.:basicsr_modified CUDA_VISIBLE_DEVICES=0 python3 corun_colabator/simple_test.py --opt dehazing_options/valid_corun.yml --input_dir Datasets/ihaze/I-HAZE/test/hazy --result_dir results/ihaze_baseline --weights CORUN+.pth --dataset ihaze
PYTHONPATH=.:basicsr_modified CUDA_VISIBLE_DEVICES=0 python3 corun_colabator/simple_test.py --opt dehazing_options/valid_corun.yml --input_dir Datasets/ihaze/I-HAZE/test/hazy --result_dir results/ihaze_lucid --weights experiments/trainS2_CORUN_RIDCP500/models/net_g_5000.pth --dataset ihaze

echo "Starting Inference on O-Haze..."
PYTHONPATH=.:basicsr_modified CUDA_VISIBLE_DEVICES=0 python3 corun_colabator/simple_test.py --opt dehazing_options/valid_corun.yml --input_dir Datasets/ohaze/O-HAZY/hazy --result_dir results/ohaze_baseline --weights CORUN+.pth --dataset ohaze
PYTHONPATH=.:basicsr_modified CUDA_VISIBLE_DEVICES=0 python3 corun_colabator/simple_test.py --opt dehazing_options/valid_corun.yml --input_dir Datasets/ohaze/O-HAZY/hazy --result_dir results/ohaze_lucid --weights experiments/trainS2_CORUN_RIDCP500/models/net_g_5000.pth --dataset ohaze

echo "Starting Evaluations..."
rm -f eval_results.txt

echo "=== I-Haze Baseline ===" >> eval_results.txt
PYTHONPATH=.:basicsr_modified CUDA_VISIBLE_DEVICES=0 python evaluate_full.py --input_dir results/ihaze_baseline --gt_dir Datasets/ihaze/I-HAZE/test/clear >> eval_results.txt 2>&1

echo "=== I-Haze LUCID ===" >> eval_results.txt
PYTHONPATH=.:basicsr_modified CUDA_VISIBLE_DEVICES=0 python evaluate_full.py --input_dir results/ihaze_lucid --gt_dir Datasets/ihaze/I-HAZE/test/clear >> eval_results.txt 2>&1

echo "=== O-Haze Baseline ===" >> eval_results.txt
PYTHONPATH=.:basicsr_modified CUDA_VISIBLE_DEVICES=0 python evaluate_full.py --input_dir results/ohaze_baseline --gt_dir Datasets/ohaze/O-HAZY/GT >> eval_results.txt 2>&1

echo "=== O-Haze LUCID ===" >> eval_results.txt
PYTHONPATH=.:basicsr_modified CUDA_VISIBLE_DEVICES=0 python evaluate_full.py --input_dir results/ohaze_lucid --gt_dir Datasets/ohaze/O-HAZY/GT >> eval_results.txt 2>&1

echo "Evaluations complete!"
