DeepMoIC Accuracy-Focused Final Branch
=====================================

Main idea
---------
This branch is not a simple layer-count change. It upgrades the old DeepMoIC-style code in five ways:

1. Validation-selected training
   - Each outer split is divided into train/validation.
   - The best epoch is selected by validation macro-F1, not training score.

2. Refit-on-full-training strategy
   - After the best epoch is selected, the model is retrained on the full outer training set for that epoch count.
   - The test set remains untouched until final evaluation.

3. New fusion autoencoder
   - ReLU autoencoder option.
   - View-attention fusion to learn which omics view contributes more.
   - Dropout and LayerNorm for more stable latent representations.

4. Dataset-specific architecture selection
   - LGG: lighter baseline GCN with ReLU AE and sqrt-inverse class weighting.
   - BRCA: stronger baseline GCN with attention fusion and sqrt-inverse weighting.
   - KIPAN: new dual-path gated model.

5. Dual-path gated architecture for graph-friendly datasets
   - Graph path: GCNII-style graph branch.
   - Direct path: MLP branch from fused omics latent representation.
   - Gate: sample-wise gate decides how much to trust graph vs direct omics features.

How to run
----------
Copy your data folder into this directory, then:

conda activate torch_env
chmod +x *.sh
bash run_acc_all_80_20.sh

Run one dataset:
bash run_acc_lgg_80_20.sh
bash run_acc_brca_80_20.sh
bash run_acc_kipan_80_20.sh

Optional comparison:
bash run_acc_all_60_40.sh

Optional ablation:
bash run_lgg_direct_mlp_80_20.sh
bash run_brca_dual_path_80_20.sh

What to tell in board
---------------------
I upgraded the original DeepMoIC implementation by adding leakage-controlled validation-based model selection,
refit-on-full-training evaluation, attention-based omics fusion, dataset-specific architecture selection,
and a dual-path gated graph/direct classifier. The goal is to avoid forcing one GCN structure on all cancer cohorts.
