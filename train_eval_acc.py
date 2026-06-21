import inspect
# =========================
# IMPORTS
# =========================

import json
import os
import random
import time

from dataclasses import asdict, replace
from typing import List

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)

from config_acc import Config

from data_utils_acc import (
    MultiViewStandardizer,
    apply_brca_feature_selection,
    compute_metrics,
    load_dataset,
    save_confusion_matrix_csv,
)

from graph_utils_acc import (
    build_train_test_graph_from_views,
    build_train_test_graph_from_views_cosine,
    build_transductive_graph_from_views,
)

from losses_acc import (
    classification_loss,
    supervised_graph_contrastive_loss,
)

from models_acc import (
    BaselineGCNClassifier,
    BRCADualPathSubtypeAwareClassifier,
    ChebNetClassifier,
    DualPathGatedClassifier,
    HybridPrototypeClassifier,
    MultiOmicsAE,
)

# =========================
# DEFAULT PAPER METRICS
# =========================

DEFAULT_PAPER_METRICS = {
    "LGG": {},
    "KIPAN": {},
    "BRCA": {},
}


# =========================
# SEED
# =========================

def set_seed(seed: int):
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# DATASET DEFAULTS
# =========================

def apply_dataset_defaults(cfg: Config) -> Config:

    ds = cfg.dataset.upper()

    base = Config()

    def set_if_base(name, value):
        if getattr(cfg, name) == getattr(base, name):
            setattr(cfg, name, value)

    # -------------------------
    # AUTO ARCHITECTURE
    # -------------------------

    if cfg.architecture == "auto":

        if ds == "KIPAN":
            cfg.architecture = "dual_path_gated"

        elif ds == "BRCA":
            cfg.architecture = "brca_hybrid_dual_path"

        else:
            cfg.architecture = "baseline_gcn"

    # =========================
    # KIPAN
    # =========================

    if ds == "KIPAN":

        set_if_base("ae_activation", "relu")
        set_if_base("ae_hidden_dim", 128)
        set_if_base("fused_latent_dim", 128)
        set_if_base("ae_dropout", 0.05)

        set_if_base("gcn_layers", 16)
        set_if_base("gcn_hidden_dim", 96)
        set_if_base("gcn_dropout", 0.20)

        set_if_base("class_weight_mode", "sqrt_inverse")

        set_if_base("fusion_mode", "attention")

        set_if_base("mlp_hidden_dim", 128)

        set_if_base("direct_dropout", 0.25)

        set_if_base("loss_type", "focal")

        set_if_base("focal_gamma", 1.5)

        set_if_base("ensemble_size", 4)

        set_if_base("dynamic_graph_refine", False)

        set_if_base("dynamic_graph_k", 20)

        set_if_base("dynamic_graph_blend", 0.35)

        set_if_base("label_smoothing", 0.05)

        set_if_base("contrastive_loss_weight", 0.10)

        set_if_base("gcn_epochs", 300)

        set_if_base("patience", 30)

        set_if_base("fusion_hidden_dim", 64)

        set_if_base("ae_epochs", 120)

    # =========================
    # BRCA
    # =========================

    elif ds == "BRCA":

        set_if_base("split_mode", "repeated_holdout")

        set_if_base("ae_activation", "relu")
        set_if_base("ae_hidden_dim", 128)
        set_if_base("fused_latent_dim", 128)
        set_if_base("ae_dropout", 0.05)

        set_if_base("gcn_layers", 2)
        set_if_base("gcn_hidden_dim", 64)
        set_if_base("gcn_dropout", 0.10)

        set_if_base("gcn_lr", 1e-3)

        set_if_base("class_weight_mode", "sqrt_inverse")

        set_if_base("fusion_mode", "attention")

        set_if_base("direct_hidden_dim", 128)

        set_if_base("mlp_hidden_dim", 128)

        set_if_base("direct_dropout", 0.20)

        set_if_base("loss_type", "cross_entropy")

        set_if_base("ensemble_size", 5)

        set_if_base("dynamic_graph_refine", True)

        set_if_base("dynamic_graph_k", 5)

        set_if_base("dynamic_graph_blend", 0.05)

        set_if_base("patience", 35)

        set_if_base("fusion_hidden_dim", 64)

        set_if_base("graph_k_candidates", (5,))

        set_if_base("graph_similarity_threshold", 0.15)

        set_if_base("gcn_epochs", 300)

        set_if_base("label_smoothing", 0.1)

        set_if_base("brca_feature_select_k", 300)

        set_if_base("prototype_temperature", 0.7)

        set_if_base("contrastive_loss_weight", 0.15)

        set_if_base(
            "contrastive_target_classes",
            ("LumA", "LumB", "Her2"),
        )

    # =========================
    # LGG
    # =========================

    elif ds == "LGG":

        set_if_base("ae_activation", "relu")

        set_if_base("ae_hidden_dim", 128)

        set_if_base("fused_latent_dim", 96)

        set_if_base("ae_dropout", 0.05)

        set_if_base("gcn_layers", 8)

        set_if_base("gcn_hidden_dim", 64)

        set_if_base("gcn_dropout", 0.25)

        set_if_base("class_weight_mode", "sqrt_inverse")

        set_if_base("fusion_mode", "attention")

        set_if_base("loss_type", "cross_entropy")

        set_if_base("ensemble_size", 5)

        set_if_base("contrastive_loss_weight", 0.0)

        set_if_base("binary_threshold_tuning", False)

        set_if_base("min_final_epochs", 100)

        set_if_base("dynamic_graph_refine", True)

        set_if_base("dynamic_graph_k", 20)

        set_if_base("dynamic_graph_blend", 0.35)

        set_if_base("patience", 35)

        set_if_base("gcn_epochs", 300)

    return cfg


@torch.no_grad()
def encode_views(model: MultiOmicsAE, views: List[np.ndarray], device: torch.device):
    model.eval()
    x = [torch.tensor(v, dtype=torch.float32, device=device) for v in views]
    z, _, _, weights = model(x[0], x[1], x[2])
    attention = None if weights is None else weights.cpu().numpy().astype(np.float32)
    return z.cpu().numpy().astype(np.float32), attention


def fit_autoencoder(train_views: List[np.ndarray], cfg: Config, device: torch.device):
    x = [torch.tensor(v, dtype=torch.float32, device=device) for v in train_views]
    model = MultiOmicsAE(
        d1=x[0].shape[1], d2=x[1].shape[1], d3=x[2].shape[1],
        hidden_dim=cfg.ae_hidden_dim, latent_per_view=cfg.ae_latent_per_view,
        fused_latent_dim=cfg.fused_latent_dim, activation=cfg.ae_activation,
        dropout=cfg.ae_dropout, fusion_mode=cfg.fusion_mode,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ae_lr, weight_decay=1e-6)
    hist = []
    for _ in range(cfg.ae_epochs):
        model.train(); opt.zero_grad()
        _, _, rec, _ = model(x[0], x[1], x[2])
        loss = sum(torch.nn.functional.mse_loss(r, v) for r, v in zip(rec, x)) / 3.0
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
    return model, hist


def make_class_weights(y_train: np.ndarray, num_classes: int, mode: str, device):
    if mode == "none":
        return None
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    if mode == "inverse":
        w = 1.0 / counts
    elif mode == "sqrt_inverse":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class_weight_mode: {mode}")
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def build_classifier(cfg: Config, num_classes: int):
    if cfg.architecture == "brca_hybrid_dual_path":
        return BRCADualPathSubtypeAwareClassifier(
            input_dim=cfg.fused_latent_dim,
            gcn_hidden_dim=cfg.gcn_hidden_dim,
            direct_hidden_dim=cfg.direct_hidden_dim,
            num_classes=num_classes,
            num_layers=cfg.gcn_layers,
            K=3,
            gcn_dropout=cfg.gcn_dropout,
            direct_dropout=cfg.direct_dropout,
            prototype_temperature=cfg.prototype_temperature,
            dynamic_graph_refine=cfg.dynamic_graph_refine,
            dynamic_graph_k=cfg.dynamic_graph_k,
            dynamic_graph_blend=cfg.dynamic_graph_blend,
        )
    if cfg.architecture == "dual_path_gated":
        return DualPathGatedClassifier(
            cfg.fused_latent_dim, cfg.gcn_hidden_dim, cfg.mlp_hidden_dim, cfg.fusion_hidden_dim,
            num_classes, cfg.gcn_layers, cfg.gcn_alpha, cfg.gcn_theta, cfg.gcn_dropout, cfg.direct_dropout,
        )
    if cfg.architecture == "hybrid_proto":
        return HybridPrototypeClassifier(
            input_dim=cfg.fused_latent_dim,
            hidden_dim=cfg.gcn_hidden_dim,
            fusion_hidden_dim=cfg.fusion_hidden_dim,
            num_classes=num_classes,
            num_layers=cfg.gcn_layers,
            alpha=cfg.gcn_alpha,
            theta=cfg.gcn_theta,
            dropout=cfg.gcn_dropout,
            transformer_heads=cfg.transformer_heads,
            transformer_layers=cfg.transformer_layers,
            transformer_ff_dim=cfg.transformer_ff_dim,
            transformer_tokens=cfg.transformer_tokens,
            prototype_temperature=cfg.prototype_temperature,
            dynamic_graph_refine=cfg.dynamic_graph_refine,
            dynamic_graph_k=cfg.dynamic_graph_k,
            dynamic_graph_blend=cfg.dynamic_graph_blend,
        )
    if cfg.architecture == "chebnet":
        return ChebNetClassifier(
            cfg.fused_latent_dim,
            cfg.gcn_hidden_dim,
            num_classes,
            num_layers=max(2, cfg.gcn_layers),
            K=3,
            dropout=cfg.gcn_dropout,
            dynamic_graph_refine=cfg.dynamic_graph_refine,
            dynamic_graph_k=cfg.dynamic_graph_k,
            dynamic_graph_blend=cfg.dynamic_graph_blend,
        )
    return BaselineGCNClassifier(cfg.fused_latent_dim, cfg.gcn_hidden_dim, num_classes, cfg.gcn_layers, cfg.gcn_alpha, cfg.gcn_theta, cfg.gcn_dropout)


def prepare_features(
    train_views_raw,
    eval_views_raw,
    y_train,
    cfg: Config,
    device,
    label_names=None,
    brca_top_k=None,
):

    selected_counts = [int(v.shape[1]) for v in train_views_raw]

    # =========================================================
    # BRCA FEATURE SELECTION
    # =========================================================
    if cfg.dataset.upper() == "BRCA":

        top_k = (
            cfg.brca_feature_select_k
            if brca_top_k is None
            else brca_top_k
        )

        train_views_raw, eval_views_raw, selected_counts = (
            apply_brca_feature_selection(
                train_views_raw,
                eval_views_raw,
                y_train,
                top_k,
                cfg.brca_feature_variance_threshold,
            )
        )

    # =========================================================
    # STANDARDIZATION
    # =========================================================
    scaler = MultiViewStandardizer().fit(train_views_raw)

    train_views = scaler.transform(train_views_raw)
    eval_views = scaler.transform(eval_views_raw)

    # =========================================================
    # AUTOENCODER
    # =========================================================
    t0 = time.time()

    ae, ae_hist = fit_autoencoder(train_views, cfg, device)

    ae_time = time.time() - t0

    # =========================================================
    # LATENT REPRESENTATIONS
    # =========================================================
    z_train, attn_train = encode_views(ae, train_views, device)

    z_eval, attn_eval = encode_views(ae, eval_views, device)

    # =========================================================
    # NORMALIZATION
    # =========================================================
    mu = z_train.mean(axis=0, keepdims=True)

    sig = z_train.std(axis=0, keepdims=True) + 1e-8

    z_train = (z_train - mu) / sig

    z_eval = (z_eval - mu) / sig

    # =========================================================
    # GRAPH CONSTRUCTION
    # =========================================================

    if cfg.dataset.upper() == "BRCA":

        adj = build_train_test_graph_from_views_cosine(
            train_views,
            eval_views,
            k=cfg.graph_k_candidates[0],
            threshold=cfg.graph_similarity_threshold,
            self_loop=cfg.graph_self_loop,
        )

    elif cfg.dataset.upper() == "KIPAN":

        adj = build_transductive_graph_from_views(
            train_views,
            eval_views,
            cfg.snf_k,
            cfg.snf_mu,
            cfg.edge_dropout,
            cfg.random_state,
        )

    elif cfg.graph_mode == "transductive":

        adj = build_transductive_graph_from_views(
            train_views,
            eval_views,
            cfg.snf_k,
            cfg.snf_mu,
            cfg.edge_dropout,
            cfg.random_state,
        )

    else:

        adj = build_train_test_graph_from_views(
            train_views,
            eval_views,
            cfg.snf_k,
            cfg.snf_mu,
            cfg.cross_k,
            cfg.edge_dropout,
            cfg.random_state,
        )

    # =========================================================
    # ATTENTION
    # =========================================================
    attention_mean = None

    if attn_train is not None:

        attention_mean = (
            np.vstack([attn_train, attn_eval])
            .mean(axis=0)
            .tolist()
        )

    # =========================================================
    # RETURN
    # =========================================================
    return (
        z_train.astype(np.float32),
        z_eval.astype(np.float32),
        adj.astype(np.float32),
        ae_hist,
        ae_time,
        selected_counts,
        attention_mean,
    )


def select_best_graph_k(
    train_views_raw,
    val_views_raw,
    z_train,
    z_val,
    y_train,
    y_val,
    cfg: Config,
    label_names,
    device,
):
    best_score = -1.0
    best_k = cfg.graph_k_candidates[0]
    best_adj = None
    for k in cfg.graph_k_candidates:

        if cfg.dataset.upper() == "KIPAN":

            adj_val = build_transductive_graph_from_views(
                train_views_raw,
                val_views_raw,
                snf_k=k,
                snf_mu=cfg.snf_mu,
                edge_dropout=cfg.edge_dropout,
                seed=cfg.random_state,
            )

        else:

            adj_val = build_train_test_graph_from_views(
                train_views_raw,
                val_views_raw,
                snf_k=k,
                snf_mu=cfg.snf_mu,
                cross_k=cfg.cross_k,
                edge_dropout=cfg.edge_dropout,
                seed=cfg.random_state,
            )
        _, _, _, _, score, _, _ = train_model_on_graph(
            z_train,
            z_val,
            adj_val,
            y_train,
            cfg,
            len(label_names),
            device,
            label_names,
            val_y=y_val,
            select_by_val=True,
            checkpoint_path=None,
        )
        if score > best_score:
            best_score = score
            best_k = k
            best_adj = adj_val
    return best_k, best_adj, best_score


def search_brca_hyperparams(
    train_views_raw,
    val_views_raw,
    y_train,
    y_val,
    cfg: Config,
    label_names,
    device,
):
    best_score = -1.0
    best_result = None
    for feature_k in cfg.brca_feature_select_k_candidates:
        z_train, z_val, adj_val, _, _, selected_counts, attention_mean = prepare_features(
            train_views_raw,
            val_views_raw,
            y_train,
            cfg,
            device,
            label_names,
            brca_top_k=feature_k,
        )
        for hidden_dim in cfg.brca_gcn_hidden_dim_candidates:
            for dropout in cfg.brca_gcn_dropout_candidates:
                for lr in cfg.brca_gcn_lr_candidates:
                    for layers in cfg.brca_gcn_layers_candidates:
                        temp_cfg = replace(
                            cfg,
                            gcn_hidden_dim=hidden_dim,
                            gcn_dropout=dropout,
                            gcn_lr=lr,
                            gcn_layers=layers,
                        )
                        for graph_k in cfg.graph_k_candidates:
                            _, _, _, chosen_epoch, score, val_hist, _ = train_model_on_graph(
                                z_train,
                                z_val,
                                adj_val,
                                y_train,
                                temp_cfg,
                                len(label_names),
                                device,
                                label_names,
                                val_y=y_val,
                                select_by_val=True,
                                max_epochs=cfg.hyperparam_search_epochs,
                                checkpoint_path=None,
                            )
                            if score > best_score or (
                                score == best_score
                                and (
                                    feature_k < best_result["feature_k"]
                                    or graph_k < best_result["graph_k"]
                                    or hidden_dim < best_result["hidden_dim"]
                                    or dropout < best_result["dropout"]
                                    or lr < best_result["lr"]
                                    or layers < best_result["layers"]
                                )
                            ):
                                best_score = score
                                best_result = {
                                    "best_cfg": temp_cfg,
                                    "feature_k": feature_k,
                                    "graph_k": graph_k,
                                    "hidden_dim": hidden_dim,
                                    "dropout": dropout,
                                    "lr": lr,
                                    "layers": layers,
                                    "chosen_epoch": chosen_epoch,
                                    "best_score": float(score),
                                    "selected_counts": selected_counts,
                                    "attention_mean": attention_mean,
                                    "best_adj_val": adj_val,
                                    "best_val_hist": val_hist,
                                }
    return best_result


def _forward_logits(model, x_full, adj_t):
    out = model(x_full, adj_t)
    return out if isinstance(out, dict) else {"logits": out, "embedding": out}


def _contrastive_nodes_and_labels(out, train_nodes, y_train_t, label_names, cfg: Config):
    if cfg.contrastive_loss_weight <= 0 or "embedding" not in out:
        return None, None
    targets = {str(name).lower() for name in cfg.contrastive_target_classes}
    target_ids = [i for i, name in enumerate(label_names) if str(name).lower() in targets]
    if not target_ids:
        return None, None
    mask = torch.zeros_like(y_train_t, dtype=torch.bool)
    for class_id in target_ids:
        mask |= y_train_t.eq(class_id)
    if mask.sum() < 2:
        return None, None
    return out["embedding"][train_nodes][mask], y_train_t[mask]


def _format_subtype_f1(metrics, label_names):
    return " ".join(f"{name}:{metrics.get(f'f1_{name}', 0.0):.3f}" for name in label_names)


def _class_recall(y_true, y_pred, label_names, class_name: str):
    if class_name not in label_names:
        return 0.0
    class_id = label_names.index(class_name)
    mask = np.asarray(y_true) == class_id
    if not mask.any():
        return 0.0
    return float(np.mean(np.asarray(y_pred)[mask] == class_id))


def train_model_on_graph(z_train, z_eval, adj, y_train, cfg: Config, num_classes: int, device, label_names, val_y=None, max_epochs=None, select_by_val=False, checkpoint_path=None, split_id=None, member_id=None):
    x_full = torch.tensor(np.vstack([z_train, z_eval]), dtype=torch.float32, device=device)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    n_train = len(z_train)
    train_nodes = torch.arange(n_train, device=device)
    eval_nodes = torch.arange(n_train, n_train + len(z_eval), device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    y_full = torch.full((len(z_train) + len(z_eval),), -1, dtype=torch.long, device=device)
    y_full[:n_train] = y_train_t
    weights = make_class_weights(y_train, num_classes, cfg.class_weight_mode, device)
    model = build_classifier(cfg, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.gcn_lr, weight_decay=cfg.gcn_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=cfg.lr_scheduler_factor, patience=cfg.lr_scheduler_patience)
    max_epochs = int(max_epochs or cfg.gcn_epochs)
    best_state = None; best_score = -1.0; best_epoch = 0; bad = 0; hist = []
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train(); opt.zero_grad()
        out = _forward_logits(model, x_full, adj_t)
        logits = out["logits"]
        clf = classification_loss(
            logits[train_nodes],
            y_train_t,
            weight=weights,
            label_smoothing=cfg.label_smoothing,
            loss_type=cfg.loss_type,
            focal_gamma=cfg.focal_gamma,
        )
        contrastive = torch.tensor(0.0, device=device)
        contrastive_embeddings, contrastive_labels = _contrastive_nodes_and_labels(out, train_nodes, y_train_t, label_names, cfg)
        if contrastive_embeddings is not None:
            contrastive = supervised_graph_contrastive_loss(
                contrastive_embeddings,
                contrastive_labels,
                temperature=cfg.contrastive_temperature,
            )
        loss = cfg.classification_loss_weight * clf + cfg.contrastive_loss_weight * contrastive
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            eval_out = _forward_logits(model, x_full, adj_t)
            if select_by_val and val_y is not None:
                pred = eval_out["logits"][eval_nodes].argmax(dim=1).cpu().numpy()
                eval_metrics = compute_metrics(val_y, pred, label_names)
                lumb_recall = _class_recall(val_y, pred, label_names, "LumB")
                score = eval_metrics["macro_f1"]
                score_label = "Val Macro-F1"
            else:
                pred = eval_out["logits"][train_nodes].argmax(dim=1).cpu().numpy()
                eval_metrics = compute_metrics(y_train, pred, label_names)
                lumb_recall = _class_recall(y_train, pred, label_names, "LumB")
                score = eval_metrics["macro_f1"]
                score_label = "Train Macro-F1"
        scheduler.step(score)
        lr = opt.param_groups[0]["lr"]
        row = {"epoch": epoch + 1, "loss": float(loss.item()), "classification_loss": float(clf.item()), "contrastive_loss": float(contrastive.item()), "selection_macro_f1": float(score), "lumb_recall": float(lumb_recall), "lr": float(lr)}
        for name in label_names:
            row[f"f1_{name}"] = float(eval_metrics.get(f"f1_{name}", 0.0))
        hist.append(row)
        if cfg.dataset.upper() == "BRCA":
            fold_text = "?" if split_id is None else str(split_id)
            member_text = "val" if member_id is None else str(member_id + 1)
            print(
                f"[Fold {fold_text} | Member {member_text} | Epoch {epoch + 1}/{max_epochs}] "
                f"loss={loss.item():.4f} ce={clf.item():.4f} supcon={contrastive.item():.4f} "
                f"{score_label}={score:.4f} subtype_f1=({_format_subtype_f1(eval_metrics, label_names)}) "
                f"LumB_recall={lumb_recall:.4f} lr={lr:.6f} best={best_score:.4f}"
            )
        if score > best_score + cfg.early_stop_min_delta:
            best_score = score; best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    gcn_time = time.time() - t0
    if cfg.dataset.upper() == "BRCA":
        print(f"[Training Complete] Best epoch: {best_epoch} Best Val Macro-F1: {best_score:.4f} LR final: {opt.param_groups[0]['lr']:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = _forward_logits(model, x_full, adj_t)
        probs = torch.softmax(out["logits"][eval_nodes], dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)
    return model, probs, pred, best_epoch, best_score, hist, gcn_time


def run_final_member(member_id, split_id, train_idx, test_idx, views, y, cfg, label_names, device, chosen_epoch, chosen_k, brca_feature_k=None):
    set_seed(cfg.random_state + split_id * 100 + member_id)
    train_views_raw = [v[train_idx] for v in views]
    test_views_raw = [v[test_idx] for v in views]
    z_train, z_test, adj_test,ae_hist, ae_time, selected_counts, attn = prepare_features(
        train_views_raw,
        test_views_raw,
        y[train_idx],
        cfg,
        device,
        label_names,
        brca_top_k=brca_feature_k,
    )
    _, probs, preds, _, _, final_hist, gcn_time = train_model_on_graph(
        z_train,
        z_test,
        adj_test,
        y[train_idx],
        cfg,
        len(label_names),
        device,
        label_names,
        max_epochs=chosen_epoch,
        select_by_val=False,
        checkpoint_path=os.path.join(cfg.out_dir, f"split_{split_id}", f"member_{member_id}_best.pth"),
        split_id=split_id,
        member_id=member_id,
    )
    for row in final_hist:
        row["ensemble_member"] = member_id
    shape_log = {
        "member": member_id,
        "z_train": list(z_train.shape),
        "z_eval": list(z_test.shape),
        "adjacency": list(adj_test.shape),
        "selected_features_per_view": selected_counts,
        "attention_mean": attn,
    }
    return probs, preds, final_hist, ae_hist, ae_time, gcn_time, shape_log


def run_one_split(split_id: int, train_idx, test_idx, views, y, samples, label_names, cfg: Config, device):
    split_dir = os.path.join(cfg.out_dir, f"split_{split_id}"); os.makedirs(split_dir, exist_ok=True)
    inner_train_rel, val_rel = train_test_split(
        np.arange(len(train_idx)), test_size=cfg.val_ratio, stratify=y[train_idx], random_state=cfg.random_state + split_id
    )
    inner_train_idx = train_idx[inner_train_rel]
    val_idx = train_idx[val_rel]

    inner_views_raw = [v[inner_train_idx] for v in views]
    val_views_raw = [v[val_idx] for v in views]
    best_cfg = cfg
    chosen_feature_k = None
    if cfg.dataset.upper() == "BRCA":
        chosen_feature_k = int(cfg.brca_feature_select_k)
        chosen_k = 5
        z_inner, z_val, best_adj_val,ae_hist_val, _, selected_counts_val, attn_val = prepare_features(
            inner_views_raw,
            val_views_raw,
            y[inner_train_idx],
            cfg,
            device,
            label_names,
            brca_top_k=chosen_feature_k,
        )

        _, val_probs, _, chosen_epoch, best_val, val_hist, _ = train_model_on_graph(
            z_inner,
            z_val,
            best_adj_val,
            y[inner_train_idx],
            cfg,
            len(label_names),
            device,
            label_names,
            val_y=y[val_idx],
            select_by_val=True,
            split_id=split_id,
            member_id=None,
        )
        chosen_epoch = max(int(chosen_epoch), cfg.min_final_epochs)
    else:
        z_inner, z_val, _,ae_hist_val, _, selected_counts_val, attn_val = prepare_features(inner_views_raw, val_views_raw, y[inner_train_idx], cfg, device, label_names)
        chosen_k, best_adj_val, best_val = select_best_graph_k(
            inner_views_raw,
            val_views_raw,
            z_inner,
            z_val,
            y[inner_train_idx],
            y[val_idx],
            cfg,
            label_names,
            device,
        )
        _, val_probs, _, chosen_epoch, _, val_hist, _ = train_model_on_graph(
            z_inner,
            z_val,
            best_adj_val,
            y[inner_train_idx],
            cfg,
            len(label_names),
            device,
            label_names,
            val_y=y[val_idx],
            select_by_val=True,
            split_id=split_id,
            member_id=None,
        )
        chosen_epoch = max(int(chosen_epoch), cfg.min_final_epochs)

    ensemble_size = max(1, int(cfg.ensemble_size))
    all_probs = []
    final_hist_all = []
    ae_hist_first = None
    shape_logs = []
    ae_time_total = 0.0
    gcn_time_total = 0.0
    for member_id in range(ensemble_size):
        print(f"[Ensemble {member_id + 1}/{ensemble_size}] Starting member training")
        probs, _, final_hist, ae_hist, ae_time, gcn_time, shape_log = run_final_member(
            member_id,
            split_id,
            train_idx,
            test_idx,
            views,
            y,
            best_cfg,
            label_names,
            device,
            chosen_epoch,
            chosen_k,
            brca_feature_k=chosen_feature_k,
        )
        print(f"[Ensemble {member_id + 1}/{ensemble_size}] Completed member training")
        all_probs.append(probs)
        final_hist_all.extend(final_hist)
        ae_hist_first = ae_hist if ae_hist_first is None else ae_hist_first
        ae_time_total += ae_time
        gcn_time_total += gcn_time
        shape_logs.append(shape_log)

    probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    preds = probs.argmax(axis=1)
    metrics = compute_metrics(y[test_idx], preds, label_names)
    metrics.update({
        "split": split_id,
        "chosen_epoch_from_val": chosen_epoch,
        "chosen_feature_k": int(chosen_feature_k) if chosen_feature_k is not None else int(cfg.brca_feature_select_k),
        "chosen_graph_k": int(chosen_k),
        "chosen_hidden_dim": int(best_cfg.gcn_hidden_dim),
        "chosen_dropout": float(best_cfg.gcn_dropout),
        "chosen_lr": float(best_cfg.gcn_lr),
        "chosen_gcn_layers": int(best_cfg.gcn_layers),
        "best_val_macro_f1": float(best_val),
        "ae_time_sec": ae_time_total,
        "gcn_time_sec": gcn_time_total,
        "architecture_used": cfg.architecture,
        "ensemble_size": ensemble_size,
    })
    pred_df = pd.DataFrame({"split": split_id, "sample_id": np.array(samples)[test_idx], "y_true": y[test_idx], "y_pred": preds, "y_true_name": [label_names[i] for i in y[test_idx]], "y_pred_name": [label_names[i] for i in preds]})
    for c_idx, cname in enumerate(label_names):
        pred_df[f"prob_{cname}"] = probs[:, c_idx]
    pred_df.to_csv(os.path.join(split_dir, "test_predictions.csv"), index=False)
    pd.DataFrame(val_hist).to_csv(os.path.join(split_dir, "inner_validation_history.csv"), index=False)
    pd.DataFrame(final_hist_all).to_csv(os.path.join(split_dir, "final_train_history.csv"), index=False)
    pd.DataFrame({"ae_loss": ae_hist_first or []}).to_csv(os.path.join(split_dir, "ae_train_history.csv"), index=False)
    save_confusion_matrix_csv(os.path.join(split_dir, "confusion_matrix.csv"), y[test_idx], preds, label_names)
    with open(os.path.join(split_dir, "validation_shapes.json"), "w", encoding="utf-8") as f:
        shape_payload = {"inner": {"z_train": list(z_inner.shape), "z_eval": list(z_val.shape), "adjacency": list(best_adj_val.shape), "selected_features_per_view": selected_counts_val, "attention_mean": attn_val}, "final": shape_logs}
        json.dump(shape_payload, f, indent=2)
    return metrics, pred_df


def iter_splits(cfg: Config, y: np.ndarray):
    idx = np.arange(len(y))
    if cfg.split_mode == "kfold":
        skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        for sid, (tr, te) in enumerate(skf.split(idx, y), 1):
            yield sid, tr, te
    else:
        sss = StratifiedShuffleSplit(n_splits=cfg.repeats, test_size=cfg.test_size, random_state=cfg.random_state)
        for sid, (tr, te) in enumerate(sss.split(idx, y), 1):
            yield sid, tr, te


def architecture_summary(cfg: Config, num_classes: int):

    graph_base = "cosine similarity KNN"

    if cfg.dataset.upper() in {"KIPAN","LGG"}:
        graph_base = "SNF patient similarity"

    return {
        "dataset": cfg.dataset,
        "architecture": cfg.architecture,

        "autoencoder": {
            "activation": cfg.ae_activation,
            "dropout": cfg.ae_dropout,
            "layer_norm": True,
            "latent_per_view": cfg.ae_latent_per_view,
            "fused_latent_dim": cfg.fused_latent_dim,
            "fusion_mode": cfg.fusion_mode,
        },

        "graph": {
            "base": graph_base,
            "dynamic_refinement": cfg.dynamic_graph_refine,
            "graph_k_candidates": cfg.graph_k_candidates,
            "graph_threshold": cfg.graph_similarity_threshold,
            "dynamic_blend": cfg.dynamic_graph_blend,
            "self_loop": cfg.graph_self_loop,
        },

        "brca_search": {
            "feature_k_candidates": cfg.brca_feature_select_k_candidates,
            "variance_threshold": cfg.brca_feature_variance_threshold,
            "hidden_dim_candidates": cfg.brca_gcn_hidden_dim_candidates,
            "dropout_candidates": cfg.brca_gcn_dropout_candidates,
            "lr_candidates": cfg.brca_gcn_lr_candidates,
            "layers_candidates": cfg.brca_gcn_layers_candidates,
            "hyperparam_search_epochs": cfg.hyperparam_search_epochs,
        },

        "classifier": {
            "type": cfg.architecture,
            "num_classes": num_classes,
            "prototype_classifier": cfg.architecture in {
                "hybrid_proto",
                "brca_hybrid_dual_path"
            }
        },

        "losses": {
            "classification": cfg.loss_type,
            "classification_weight": cfg.classification_loss_weight,
            "focal_gamma": cfg.focal_gamma if cfg.loss_type == "focal" else None,
            "supervised_contrastive_weight": cfg.contrastive_loss_weight,
            "supervised_contrastive_temperature": cfg.contrastive_temperature,
            "supervised_contrastive_classes": cfg.contrastive_target_classes,
            "class_weight_mode": cfg.class_weight_mode,
        },

        "training": {
            "inner_validation": True,
            "early_stopping": True,
            "scheduler": "ReduceLROnPlateau",
            "use_cosine_annealing": cfg.use_cosine_annealing,
            "ensemble_size": cfg.ensemble_size,
        },
    }


def brca_confusion_analysis(y_true, y_pred, label_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    analysis = {"matrix": cm.tolist(), "labels": label_names, "focused_pairs": {}}
    for a, b in [("LumA", "LumB"), ("Her2", "LumB")]:
        if a in label_names and b in label_names:
            ia = label_names.index(a)
            ib = label_names.index(b)
            support_a = int(cm[ia].sum())
            support_b = int(cm[ib].sum())
            analysis["focused_pairs"][f"{a}_as_{b}"] = {
                "count": int(cm[ia, ib]),
                "rate": float(cm[ia, ib] / max(support_a, 1)),
            }
            analysis["focused_pairs"][f"{b}_as_{a}"] = {
                "count": int(cm[ib, ia]),
                "rate": float(cm[ib, ia] / max(support_b, 1)),
            }
    return analysis


def compare_to_paper(metrics_df: pd.DataFrame, cfg: Config):
    paper = cfg.paper_metrics or DEFAULT_PAPER_METRICS.get(cfg.dataset.upper(), {})
    comparison = {}
    for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
        current = float(metrics_df[metric].mean())
        paper_val = paper.get(metric)
        comparison[metric] = {"paper": paper_val, "current": current, "delta": None if paper_val is None else current - float(paper_val)}
    return comparison


def run_experiment(cfg: Config):
    cfg = apply_dataset_defaults(cfg)
    set_seed(cfg.random_state)
    os.makedirs(cfg.out_dir, exist_ok=True)
    data = load_dataset(cfg.dataset, cfg.data_root)
    views, y, samples, label_names = data["views"], data["labels"], data["samples"], data["label_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = architecture_summary(cfg, len(label_names))
    print("\n===== Architecture Summary =====")
    print(json.dumps(arch, indent=2))
    all_metrics = []; all_preds = []
    for sid, train_idx, test_idx in iter_splits(cfg, y):
        m, p = run_one_split(sid, train_idx, test_idx, views, y, samples, label_names, cfg, device)
        all_metrics.append(m); all_preds.append(p)
    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True)
    summary = {"config": asdict(cfg), "architecture_summary": arch, "num_samples": int(len(y)), "num_evals": int(len(metrics_df))}
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "ae_time_sec", "gcn_time_sec"]:
        summary[f"{metric}_mean"] = float(metrics_df[metric].mean())
        summary[f"{metric}_std"] = float(metrics_df[metric].std(ddof=0))
    summary["architecture_used"] = cfg.architecture
    summary["paper_comparison"] = compare_to_paper(metrics_df, cfg)
    summary["brca_confusion_analysis"] = brca_confusion_analysis(
        preds_df["y_true"].to_numpy(),
        preds_df["y_pred"].to_numpy(),
        label_names,
    ) if cfg.dataset.upper() == "BRCA" else None
    metrics_df.to_csv(os.path.join(cfg.out_dir, "split_metrics.csv"), index=False)
    preds_df.to_csv(os.path.join(cfg.out_dir, "all_test_predictions.csv"), index=False)
    if cfg.dataset.upper() == "BRCA":
        with open(os.path.join(cfg.out_dir, "brca_confusion_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(summary["brca_confusion_analysis"], f, indent=2)
    with open(os.path.join(cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary, metrics_df, preds_df
