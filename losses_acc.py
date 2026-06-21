import torch
import torch.nn.functional as F


def classification_loss(
    logits,
    targets,
    weight=None,
    label_smoothing: float = 0.0,
    loss_type: str = "cross_entropy",
    focal_gamma: float = 1.5,
):

    # =========================================================
    # STANDARD CROSS ENTROPY
    # =========================================================
    if loss_type == "cross_entropy":
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=label_smoothing,
        )

    # =========================================================
    # FOCAL LOSS (VIPUL VERSION - BETTER FOR KIPAN)
    # =========================================================
    if loss_type != "focal":
        raise ValueError(f"Unknown classification loss: {loss_type}")

    num_classes = logits.size(1)

    log_probs = F.log_softmax(logits, dim=1)

    probs = log_probs.exp()

    with torch.no_grad():

        target_dist = torch.zeros_like(logits)

        smooth = float(label_smoothing)

        if smooth > 0:
            target_dist.fill_(smooth / max(num_classes - 1, 1))

            target_dist.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - smooth,
            )

        else:

            target_dist.scatter_(
                1,
                targets.unsqueeze(1),
                1.0,
            )

        p_t = probs.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1).clamp_min(1e-8)

        focal_factor = (
            1.0 - p_t
        ).pow(float(focal_gamma))

    per_class_loss = -target_dist * log_probs

    if weight is not None:
        per_class_loss = (
            per_class_loss * weight.view(1, -1)
        )

    return (
        focal_factor * per_class_loss.sum(dim=1)
    ).mean()


def supervised_graph_contrastive_loss(
    embeddings,
    labels,
    temperature: float = 0.2,
):

    if embeddings is None or labels.numel() < 2:
        return torch.tensor(
            0.0,
            device=labels.device,
        )

    z = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    logits = (
        z @ z.t()
    ) / max(float(temperature), 1e-8)

    eye = torch.eye(
        labels.numel(),
        dtype=torch.bool,
        device=labels.device,
    )

    positives = (
        labels[:, None].eq(labels[None, :])
        & ~eye
    )

    valid = positives.any(dim=1)

    if not valid.any():
        return torch.tensor(
            0.0,
            device=labels.device,
        )

    logits = logits.masked_fill(
        eye,
        -1e9,
    )

    log_prob = logits - torch.logsumexp(
        logits,
        dim=1,
        keepdim=True,
    )

    per_sample = -(
        log_prob * positives.float()
    ).sum(dim=1) / positives.float().sum(dim=1).clamp_min(1.0)

    return per_sample[valid].mean()