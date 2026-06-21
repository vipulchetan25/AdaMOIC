import argparse
import json

from config_acc import Config
from train_eval_acc import run_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Accuracy-focused DeepMoIC upgraded pipeline")

    p.add_argument("--dataset", type=str, required=True, choices=["LGG", "BRCA", "KIPAN"])
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument(
        "--architecture",
        type=str,
        default="auto",
        choices=[
            "auto",
            "baseline_gcn",
            "dual_path_gated",
            "hybrid_proto",
            "brca_hybrid_dual_path",
            "chebnet",
        ],
    )

    p.add_argument(
        "--split_mode",
        type=str,
        default="repeated_holdout",
        choices=["repeated_holdout", "kfold"],
    )

    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--gcn_layers", type=int, default=None)
    p.add_argument("--ae_activation", type=str, default=None, choices=["sigmoid", "relu"])
    p.add_argument("--class_weight_mode", type=str, default=None, choices=["none", "inverse", "sqrt_inverse"])
    p.add_argument("--fusion_mode", type=str, default=None, choices=["concat", "attention", "mean"])

    # LGG graph settings
    p.add_argument("--graph_mode", type=str, default=None, choices=["inductive", "transductive"])
    p.add_argument("--edge_dropout", type=float, default=None)

    # Training
    p.add_argument("--ae_epochs", type=int, default=None)
    p.add_argument("--gcn_epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--min_final_epochs", type=int, default=None)

    # BRCA
    p.add_argument("--brca_feature_select_k", type=int, default=None)

    # Loss
    p.add_argument("--loss_type", type=str, default=None, choices=["cross_entropy", "focal"])
    p.add_argument("--focal_gamma", type=float, default=None)

    # Contrastive
    p.add_argument("--contrastive_loss_weight", type=float, default=None)
    p.add_argument("--contrastive_temperature", type=float, default=None)

    # Prototype
    p.add_argument("--prototype_temperature", type=float, default=None)

    # Ensemble
    p.add_argument("--ensemble_size", type=int, default=None)

    # Binary threshold tuning (LGG)
    p.add_argument("--binary_threshold_tuning", action="store_true")
    p.add_argument("--no_binary_threshold_tuning", action="store_true")

    p.add_argument(
        "--binary_threshold_metric",
        type=str,
        default=None,
        choices=["accuracy", "balanced_accuracy", "macro_f1"],
    )

    # Dynamic graph refinement
    p.add_argument("--dynamic_graph_refine", action="store_true")
    p.add_argument("--no_dynamic_graph_refine", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = Config(
        dataset=args.dataset,
        data_root=args.data_root,
        out_dir=args.out_dir,
    )

    cfg.architecture = args.architecture
    cfg.split_mode = args.split_mode
    cfg.test_size = args.test_size
    cfg.repeats = args.repeats
    cfg.n_splits = args.n_splits
    cfg.random_state = args.seed

    if args.gcn_layers is not None:
        cfg.gcn_layers = args.gcn_layers

    if args.ae_activation is not None:
        cfg.ae_activation = args.ae_activation

    if args.class_weight_mode is not None:
        cfg.class_weight_mode = args.class_weight_mode

    if args.fusion_mode is not None:
        cfg.fusion_mode = args.fusion_mode

    if args.graph_mode is not None:
        cfg.graph_mode = args.graph_mode

    if args.edge_dropout is not None:
        cfg.edge_dropout = args.edge_dropout

    if args.ae_epochs is not None:
        cfg.ae_epochs = args.ae_epochs

    if args.gcn_epochs is not None:
        cfg.gcn_epochs = args.gcn_epochs

    if args.patience is not None:
        cfg.patience = args.patience

    if args.min_final_epochs is not None:
        cfg.min_final_epochs = args.min_final_epochs

    if args.brca_feature_select_k is not None:
        cfg.brca_feature_select_k = args.brca_feature_select_k

    if args.loss_type is not None:
        cfg.loss_type = args.loss_type

    if args.focal_gamma is not None:
        cfg.focal_gamma = args.focal_gamma

    if args.contrastive_loss_weight is not None:
        cfg.contrastive_loss_weight = args.contrastive_loss_weight

    if args.contrastive_temperature is not None:
        cfg.contrastive_temperature = args.contrastive_temperature

    if args.prototype_temperature is not None:
        cfg.prototype_temperature = args.prototype_temperature

    if args.ensemble_size is not None:
        cfg.ensemble_size = args.ensemble_size

    if args.binary_threshold_tuning:
        cfg.binary_threshold_tuning = True

    if args.no_binary_threshold_tuning:
        cfg.binary_threshold_tuning = False

    if args.binary_threshold_metric is not None:
        cfg.binary_threshold_metric = args.binary_threshold_metric

    if args.dynamic_graph_refine:
        cfg.dynamic_graph_refine = True

    if args.no_dynamic_graph_refine:
        cfg.dynamic_graph_refine = False

    summary, metrics_df, _ = run_experiment(cfg)

    print("\n===== Final Summary =====")
    print(json.dumps(summary, indent=2))

    print("\nPer-split metrics:")
    print(metrics_df.to_string(index=False))