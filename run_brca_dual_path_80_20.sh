#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python main_deepmoic_acc.py --dataset BRCA --data_root data --out_dir outputs_BRCA_dual_path_80_20 --architecture dual_path_gated --test_size 0.2 --repeats 5 --loss_type luminal_pair --luminal_pair_penalty 3.5 --her2_lumb_pair_penalty 2.0 --use_luminal_refinement --class_weight_mode inverse
