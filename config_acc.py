from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    dataset: str = "KIPAN"
    data_root: str = "data"
    out_dir: str = "outputs_acc"
    random_state: int = 42

    # Evaluation
    split_mode: str = "repeated_holdout"   # repeated_holdout or kfold
    n_splits: int = 5
    repeats: int = 5
    test_size: float = 0.2
    val_ratio: float = 0.15

    # Autoencoder
    ae_hidden_dim: int = 128
    ae_latent_per_view: int = 64
    fused_latent_dim: int = 128
    ae_epochs: int = 120
    ae_lr: float = 1e-3
    ae_activation: str = "relu"
    ae_dropout: float = 0.10
    fusion_mode: str = "attention"  # concat, attention, mean

    # Supervised feature selection
    brca_feature_select_k: int = 300
    brca_feature_select_k_candidates: tuple = (300,)
    brca_feature_variance_threshold: float = 0.001
    brca_gcn_hidden_dim_candidates: tuple = (64,)
    brca_gcn_dropout_candidates: tuple = (0.10,)
    brca_gcn_lr_candidates: tuple = (1e-3,)
    brca_gcn_layers_candidates: tuple = (2,)
    hyperparam_search_epochs: int = 25

    # Graph model
    architecture: str = "auto"  # auto, baseline_gcn, dual_path_gated, hybrid_proto, brca_hybrid_dual_path, chebnet

    gcn_hidden_dim: int = 64
    gcn_layers: int = 2
    gcn_dropout: float = 0.10

    gcn_alpha: float = 0.5
    gcn_theta: float = 0.5

    gcn_lr: float = 1e-3
    gcn_weight_decay: float = 5e-4

    gcn_epochs: int = 300
    patience: int = 35

    # Direct path
    direct_hidden_dim: int = 128
    mlp_hidden_dim: int = 128
    fusion_hidden_dim: int = 64
    direct_dropout: float = 0.20

    # Hybrid prototype model
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 192
    transformer_tokens: int = 4

    prototype_temperature: float = 0.7

    dynamic_graph_refine: bool = True
    dynamic_graph_k: int = 5
    dynamic_graph_blend: float = 0.05

    # Optimization
    class_weight_mode: str = "sqrt_inverse"  # none, inverse, sqrt_inverse

    label_smoothing: float = 0.1

    min_final_epochs: int = 20

    loss_type: str = "cross_entropy"  # cross_entropy, focal
    focal_gamma: float = 1.5

    use_cosine_annealing: bool = False

    classification_loss_weight: float = 1.0

    contrastive_loss_weight: float = 0.15
    contrastive_temperature: float = 0.2
    contrastive_target_classes: tuple = ("LumA", "LumB", "Her2")

    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5

    early_stop_min_delta: float = 1e-5

    ensemble_size: int = 5

    # Binary threshold tuning (LGG support)
    binary_threshold_tuning: Optional[bool] = None
    binary_threshold_metric: str = "macro_f1"

    # Graph construction
    graph_k_candidates: tuple = (5,)
    graph_similarity_threshold: float = 0.15
    graph_self_loop: bool = True

    # Graph mode / SNF support
    graph_mode: str = "inductive"  # inductive, transductive

    snf_k: int = 20
    snf_mu: float = 0.5
    cross_k: int = 20

    edge_dropout: float = 0.0

    paper_metrics: dict = None

    save_plots: bool = False
