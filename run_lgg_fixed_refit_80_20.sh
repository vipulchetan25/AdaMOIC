#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python main_deepmoic_acc.py --dataset LGG --data_root data --out_dir outputs_LGG_fixed_refit_80_20 --architecture auto --test_size 0.2 --repeats 5 --no_binary_threshold_tuning --min_final_epochs 100
