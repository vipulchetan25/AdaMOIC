import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler

class MultiViewStandardizer:
    def __init__(self):
        self.scalers: List[StandardScaler] = []

    def fit(self, views: List[np.ndarray]):
        self.scalers = []
        for x in views:
            sc = StandardScaler(with_mean=True, with_std=True)
            sc.fit(x)
            self.scalers.append(sc)
        return self

    def transform(self, views: List[np.ndarray]) -> List[np.ndarray]:
        return [sc.transform(x).astype(np.float32) for sc, x in zip(self.scalers, views)]


def _align_by_sample(view1_df, view2_df, view3_df, labels_df):
    key1 = view1_df.columns[0]
    key2 = view2_df.columns[0]
    key3 = view3_df.columns[0]
    key4 = labels_df.columns[0]
    common = set(view1_df[key1]) & set(view2_df[key2]) & set(view3_df[key3]) & set(labels_df[key4])
    common = sorted(common)
    view1_df = view1_df[view1_df[key1].isin(common)].copy().sort_values(key1)
    view2_df = view2_df[view2_df[key2].isin(common)].copy().sort_values(key2)
    view3_df = view3_df[view3_df[key3].isin(common)].copy().sort_values(key3)
    labels_df = labels_df[labels_df[key4].isin(common)].copy().sort_values(key4)
    return view1_df, view2_df, view3_df, labels_df


def _load_kipan_lgg(path: str):
    # The original LGG/KIPAN matrices do not contain headers. Without header=None,
    # pandas consumes the first patient as column names and silently drops it.
    view1_df = pd.read_csv(os.path.join(path, "1.csv"), header=None)
    view2_df = pd.read_csv(os.path.join(path, "2.csv"), header=None)
    view3_df = pd.read_csv(os.path.join(path, "3.csv"), header=None)
    labels_df = pd.read_csv(os.path.join(path, "labels.csv"), header=None)
    if not (len(view1_df) == len(view2_df) == len(view3_df) == len(labels_df)):
        raise ValueError("LGG/KIPAN omics views and labels must have the same row count after header=None loading.")
    x1 = view1_df.to_numpy(dtype=np.float32)
    x2 = view2_df.to_numpy(dtype=np.float32)
    x3 = view3_df.to_numpy(dtype=np.float32)
    samples = [f"sample_{i}" for i in range(len(x1))]
    y_raw = labels_df.iloc[:, -1].astype(str).to_numpy()
    return x1, x2, x3, y_raw, samples


def _load_brca(path: str):
    view1_df = pd.read_csv(os.path.join(path, "fpkm_data.csv"))
    view2_df = pd.read_csv(os.path.join(path, "gistic_data.csv"))
    view3_df = pd.read_csv(os.path.join(path, "rppa_data.csv"))
    labels_df = pd.read_csv(os.path.join(path, "sample_class.csv"))
    view1_df, view2_df, view3_df, labels_df = _align_by_sample(view1_df, view2_df, view3_df, labels_df)
    x1 = view1_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    x2 = view2_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    x3 = view3_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    samples = view1_df.iloc[:, 0].astype(str).tolist()
    y_raw = labels_df.iloc[:, -1].astype(str).to_numpy()
    return x1, x2, x3, y_raw, samples


def load_dataset(dataset: str, data_root: str = "data") -> Dict:
    dataset = dataset.upper()
    path = os.path.join(data_root, dataset)
    if dataset == "BRCA":
        x1, x2, x3, y_raw, samples = _load_brca(path)
    elif dataset in {"KIPAN", "LGG"}:
        x1, x2, x3, y_raw, samples = _load_kipan_lgg(path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    labels_sorted = sorted(pd.unique(y_raw))
    label_map = {lab: i for i, lab in enumerate(labels_sorted)}
    y = np.array([label_map[v] for v in y_raw], dtype=np.int64)
    return {"views": [x1, x2, x3], "labels": y, "label_names": labels_sorted, "samples": samples}


def _apply_variance_threshold(
    train_views: List[np.ndarray],
    eval_views: List[np.ndarray],
    threshold: float = 0.001,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    filtered_train: List[np.ndarray] = []
    filtered_eval: List[np.ndarray] = []
    for x_tr, x_ev in zip(train_views, eval_views):
        if x_tr.shape[1] <= 1:
            filtered_train.append(x_tr.astype(np.float32))
            filtered_eval.append(x_ev.astype(np.float32))
            continue
        selector = VarianceThreshold(threshold=threshold)
        try:
            x_tr_sel = selector.fit_transform(x_tr)
            if x_tr_sel.shape[1] == 0:
                x_tr_sel = x_tr[:, :1].astype(np.float32)
                x_ev_sel = x_ev[:, :1].astype(np.float32)
            else:
                x_tr_sel = x_tr_sel.astype(np.float32)
                x_ev_sel = x_ev[:, selector.get_support(indices=True)].astype(np.float32)
        except ValueError:
            x_tr_sel = x_tr.astype(np.float32)
            x_ev_sel = x_ev.astype(np.float32)
        filtered_train.append(x_tr_sel)
        filtered_eval.append(x_ev_sel)
    return filtered_train, filtered_eval

class SupervisedMISelector:
    """Per-view MI selector fit only on training data to avoid leakage."""
    def __init__(self, top_k: Optional[int]):
        self.top_k = top_k
        self.indices_: List[np.ndarray] = []

    def _safe_mutual_info(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.shape[1] == 0 or len(np.unique(y)) < 2:
            return np.zeros(x.shape[1], dtype=np.float32)
        scores = mutual_info_classif(x, y, discrete_features=False, random_state=0)
        return np.nan_to_num(scores, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)

    def fit(self, views: List[np.ndarray], y: np.ndarray):
        self.indices_ = []
        for x in views:
            k = x.shape[1] if self.top_k is None or self.top_k <= 0 else min(int(self.top_k), x.shape[1])
            if k >= x.shape[1]:
                self.indices_.append(np.arange(x.shape[1]))
                continue
            scores = self._safe_mutual_info(x, y)
            idx = np.argsort(scores)[-k:]
            self.indices_.append(np.sort(idx))
        return self

    def transform(self, views: List[np.ndarray]) -> List[np.ndarray]:
        if not self.indices_:
            return views
        return [x[:, idx].astype(np.float32) for x, idx in zip(views, self.indices_)]

    def selected_counts(self) -> List[int]:
        return [int(len(idx)) for idx in self.indices_]


def apply_brca_feature_selection(
    train_views: List[np.ndarray],
    eval_views: List[np.ndarray],
    y_train: np.ndarray,
    top_k: Optional[int],
    variance_threshold: float = 0.001,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    train_views, eval_views = _apply_variance_threshold(train_views, eval_views, threshold=variance_threshold)
    selector = SupervisedMISelector(top_k).fit(train_views, y_train)
    return selector.transform(train_views), selector.transform(eval_views), selector.selected_counts()


def apply_brca_anova_selection(
    train_views: List[np.ndarray],
    eval_views: List[np.ndarray],
    y_train: np.ndarray,
    top_k: Optional[int],
    label_names: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    return apply_brca_feature_selection(train_views, eval_views, y_train, top_k)


def compute_metrics(y_true, y_pred, label_names) -> Dict[str, float]:
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    for i, cname in enumerate(label_names):
        metrics[f"f1_{cname}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0
    return metrics


def save_confusion_matrix_csv(path: str, y_true, y_pred, label_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    df = pd.DataFrame(cm, index=[f"true_{x}" for x in label_names], columns=[f"pred_{x}" for x in label_names])
    df.to_csv(path)


def save_classification_report_txt(path: str, y_true, y_pred, label_names):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return report
