#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python main_deepmoic_acc.py --dataset LGG --data_root data --out_dir outputs_LGG_acc_80_20 --architecture auto --test_size 0.2 --repeats 5
python main_deepmoic_acc.py --dataset BRCA --data_root data --out_dir outputs_BRCA_acc_80_20 --architecture auto --test_size 0.2 --repeats 5
python main_deepmoic_acc.py --dataset KIPAN --data_root data --out_dir outputs_KIPAN_acc_80_20 --architecture auto --test_size 0.2 --repeats 5

