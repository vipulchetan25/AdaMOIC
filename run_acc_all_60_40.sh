#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python main_deepmoic_acc.py --dataset LGG --data_root data --out_dir outputs_LGG_acc_60_40 --architecture auto --test_size 0.4 --repeats 5
python main_deepmoic_acc.py --dataset BRCA --data_root data --out_dir outputs_BRCA_acc_60_40 --architecture auto --test_size 0.4 --repeats 5
python main_deepmoic_acc.py --dataset KIPAN --data_root data --out_dir outputs_KIPAN_acc_60_40 --architecture auto --test_size 0.4 --repeats 5
