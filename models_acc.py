import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_utils_acc import refine_adjacency_from_embeddings

class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.Sigmoid
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            act(),
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.Sigmoid
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, z):
        return self.net(z)

class MultiOmicsAE(nn.Module):
    """Multi-view AE with optional attention fusion over omics latent vectors."""
    def __init__(self, d1, d2, d3, hidden_dim=128, latent_per_view=64, fused_latent_dim=128, activation="relu", dropout=0.1, fusion_mode="attention"):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.enc1 = MLPBlock(d1, hidden_dim, latent_per_view, activation, dropout)
        self.enc2 = MLPBlock(d2, hidden_dim, latent_per_view, activation, dropout)
        self.enc3 = MLPBlock(d3, hidden_dim, latent_per_view, activation, dropout)
        self.dec1 = Decoder(latent_per_view, hidden_dim, d1, activation, dropout)
        self.dec2 = Decoder(latent_per_view, hidden_dim, d2, activation, dropout)
        self.dec3 = Decoder(latent_per_view, hidden_dim, d3, activation, dropout)
        if fusion_mode == "attention":
            self.view_scorer = nn.Sequential(nn.Linear(latent_per_view, max(16, latent_per_view // 2)), nn.Tanh(), nn.Linear(max(16, latent_per_view // 2), 1))
            self.fuse = nn.Linear(latent_per_view, fused_latent_dim)
        elif fusion_mode == "mean":
            self.fuse = nn.Linear(latent_per_view, fused_latent_dim)
        else:
            self.fuse = nn.Linear(latent_per_view * 3, fused_latent_dim)

    def forward(self, x1, x2, x3):
        z1, z2, z3 = self.enc1(x1), self.enc2(x2), self.enc3(x3)
        if self.fusion_mode == "attention":
            stack = torch.stack([z1, z2, z3], dim=1)  # [N,3,D]
            scores = self.view_scorer(stack).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            z_base = (stack * weights.unsqueeze(-1)).sum(dim=1)
            z_fused = self.fuse(z_base)
        elif self.fusion_mode == "mean":
            z_fused = self.fuse((z1 + z2 + z3) / 3.0)
            weights = None
        else:
            z_fused = self.fuse(torch.cat([z1, z2, z3], dim=1))
            weights = None
        r1, r2, r3 = self.dec1(z1), self.dec2(z2), self.dec3(z3)
        return z_fused, (z1, z2, z3), (r1, r2, r3), weights

class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, adj):
        return self.linear(adj @ x)

class ChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K: int = 3, dropout: float = 0.0):
        super().__init__()
        self.K = max(1, K)
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(self.K)])

    def forward(self, x, adj):
        n = x.size(0)
        I = torch.eye(n, device=adj.device, dtype=adj.dtype)
        A = adj + I
        deg = A.sum(dim=1)
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        A_norm = deg_inv_sqrt.view(-1, 1) * A * deg_inv_sqrt.view(1, -1)

        x0 = x
        out = self.linears[0](x0)
        if self.K == 1:
            return out

        x1 = A_norm @ x0
        out = out + self.linears[1](x1)
        for k in range(2, self.K):
            x2 = 2.0 * (A_norm @ x1) - x0
            out = out + self.linears[k](x2)
            x0, x1 = x1, x2
        return out

class GCNIIEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=8, alpha=0.5, theta=0.5, dropout=0.25):
        super().__init__()
        self.alpha = alpha
        self.theta = theta
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphConvolution(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, adj):
        h0 = F.relu(self.input_proj(x))
        h = self.dropout(h0)
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms), start=1):
            propagated = layer(h, adj)
            beta = math.log(self.theta / i + 1.0)
            mixed = (1 - self.alpha) * propagated + self.alpha * h0
            h = (1 - beta) * mixed + beta * h
            h = self.dropout(F.relu(norm(h)))
        return h

class BaselineGCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=8, alpha=0.5, theta=0.5, dropout=0.25):
        super().__init__()
        self.encoder = GCNIIEncoder(input_dim, hidden_dim, num_layers, alpha, theta, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, x, adj):
        h = self.encoder(x, adj)
        return {"logits": self.classifier(h), "embedding": h}

class ChebNetClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, K=3, dropout=0.25, dynamic_graph_refine=False, dynamic_graph_k=10, dynamic_graph_blend=0.25):
        super().__init__()
        self.dynamic_graph_refine = dynamic_graph_refine
        self.dynamic_graph_k = dynamic_graph_k
        self.dynamic_graph_blend = dynamic_graph_blend
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers.append(ChebConv(input_dim, hidden_dim, K=K, dropout=dropout))
        self.norms.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(ChebConv(hidden_dim, hidden_dim, K=K, dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        adj_used = adj
        if self.dynamic_graph_refine:
            adj_used = refine_adjacency_from_embeddings(adj, x, self.dynamic_graph_k, self.dynamic_graph_blend)
        h = x
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            h = conv(h, adj_used)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            if i == 0:
                h = h + self.residual(x)
        return {"logits": self.classifier(h), "embedding": h}

class DirectMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x, adj=None):
        logits = self.net(x)
        return {"logits": logits, "embedding": logits}

class DualPathGatedClassifier(nn.Module):
    """Novel branch: graph path + direct omics path + sample-wise gate."""
    def __init__(self, input_dim, gcn_hidden_dim, mlp_hidden_dim, fusion_hidden_dim, num_classes, num_layers=16, alpha=0.5, theta=0.5, gcn_dropout=0.2, direct_dropout=0.25):
        super().__init__()
        self.gcn_encoder = GCNIIEncoder(input_dim, gcn_hidden_dim, num_layers, alpha, theta, gcn_dropout)
        self.direct_encoder = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim), nn.LayerNorm(mlp_hidden_dim), nn.ReLU(), nn.Dropout(direct_dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.LayerNorm(mlp_hidden_dim), nn.ReLU(), nn.Dropout(direct_dropout),
        )
        self.g_proj = nn.Linear(gcn_hidden_dim, fusion_hidden_dim)
        self.d_proj = nn.Linear(mlp_hidden_dim, fusion_hidden_dim)
        self.gate = nn.Sequential(nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim), nn.ReLU(), nn.Linear(fusion_hidden_dim, 1), nn.Sigmoid())
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)
        self.aux_g = nn.Linear(fusion_hidden_dim, num_classes)
        self.aux_d = nn.Linear(fusion_hidden_dim, num_classes)
    def forward(self, x, adj):
        g = F.relu(self.g_proj(self.gcn_encoder(x, adj)))
        d = F.relu(self.d_proj(self.direct_encoder(x)))
        w = self.gate(torch.cat([g, d], dim=1))
        fused = w * g + (1.0 - w) * d
        logits = self.classifier(fused)
        return logits


class PrototypeClassifier(nn.Module):
    """Learnable class prototypes with temperature-scaled cosine logits."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.prototypes = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )
        self.temperature = temperature
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)

        return (x_norm @ proto_norm.t()) / max(float(self.temperature), 1e-8)
    
    
class ChebGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, K=3, dropout=0.15, dynamic_graph_refine=True, dynamic_graph_k=5, dynamic_graph_blend=0.10):
        super().__init__()
        self.dynamic_graph_refine = dynamic_graph_refine
        self.dynamic_graph_k = dynamic_graph_k
        self.dynamic_graph_blend = dynamic_graph_blend
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers.append(ChebConv(input_dim, hidden_dim, K=K, dropout=dropout))
        self.norms.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(max(1, num_layers) - 1):
            self.layers.append(ChebConv(hidden_dim, hidden_dim, K=K, dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, adj):
        adj_used = refine_adjacency_from_embeddings(adj, x, self.dynamic_graph_k, self.dynamic_graph_blend) if self.dynamic_graph_refine else adj
        h = x
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            h = conv(h, adj_used)
            h = norm(h)
            h = F.gelu(h)
            h = self.dropout(h)
            if i == 0:
                h = h + self.residual(x)
        return h, adj_used


class BRCADualPathSubtypeAwareClassifier(nn.Module):
    """ChebNet graph branch + direct MLP branch + gated fusion + cosine prototypes."""
    def __init__(
        self,
        input_dim,
        gcn_hidden_dim,
        direct_hidden_dim,
        num_classes,
        num_layers=2,
        K=3,
        gcn_dropout=0.15,
        direct_dropout=0.20,
        prototype_temperature=0.7,
        dynamic_graph_refine=True,
        dynamic_graph_k=5,
        dynamic_graph_blend=0.10,
    ):
        super().__init__()
        self.graph_encoder = ChebGraphEncoder(
            input_dim=input_dim,
            hidden_dim=gcn_hidden_dim,
            num_layers=num_layers,
            K=K,
            dropout=gcn_dropout,
            dynamic_graph_refine=dynamic_graph_refine,
            dynamic_graph_k=dynamic_graph_k,
            dynamic_graph_blend=dynamic_graph_blend,
        )
        self.direct_encoder = nn.Sequential(
            nn.Linear(input_dim, direct_hidden_dim),
            nn.BatchNorm1d(direct_hidden_dim),
            nn.GELU(),
            nn.Dropout(direct_dropout),
            nn.Linear(direct_hidden_dim, gcn_hidden_dim),
        )
        self.gate_graph = nn.Linear(gcn_hidden_dim, gcn_hidden_dim)
        self.gate_direct = nn.Linear(gcn_hidden_dim, gcn_hidden_dim)
        self.fusion_norm = nn.LayerNorm(gcn_hidden_dim)
        self.prototype_classifier = PrototypeClassifier(gcn_hidden_dim, num_classes, prototype_temperature)

    def forward(self, x, adj):
        graph_embedding, adj_used = self.graph_encoder(x, adj)
        direct_embedding = self.direct_encoder(x)
        fusion_gate = torch.sigmoid(self.gate_graph(graph_embedding) + self.gate_direct(direct_embedding))
        final_embedding = self.fusion_norm(fusion_gate * graph_embedding + (1.0 - fusion_gate) * direct_embedding)
        logits = self.prototype_classifier(final_embedding)
        return {
            "logits": logits,
            "embedding": final_embedding,
            "graph_embedding": graph_embedding,
            "direct_embedding": direct_embedding,
            "gate": fusion_gate,
            "adjacency": adj_used,
        }


class HybridPrototypeClassifier(nn.Module):
    """GCN + Transformer + dynamic graph refinement + prototype classifier."""
    def __init__(
        self,
        input_dim,
        hidden_dim,
        fusion_hidden_dim,
        num_classes,
        num_layers=8,
        alpha=0.5,
        theta=0.5,
        dropout=0.25,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff_dim=192,
        transformer_tokens=4,
        prototype_temperature=1.0,
        dynamic_graph_refine=True,
        dynamic_graph_k=20,
        dynamic_graph_blend=0.35,
    ):
        super().__init__()
        self.dynamic_graph_refine = dynamic_graph_refine
        self.dynamic_graph_k = dynamic_graph_k
        self.dynamic_graph_blend = dynamic_graph_blend
        self.gcn_encoder = GCNIIEncoder(input_dim, hidden_dim, num_layers, alpha, theta, dropout)
        self.token_count = max(1, int(transformer_tokens))
        self.token_proj = nn.Linear(input_dim, hidden_dim * self.token_count)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.graph_proj = nn.Linear(hidden_dim, fusion_hidden_dim)
        self.trans_proj = nn.Linear(hidden_dim, fusion_hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(fusion_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.prototype_classifier = PrototypeClassifier(fusion_hidden_dim, num_classes, prototype_temperature)
        self.contrastive_temperature = 0.5
        self.last_features = None

    def forward(self, x, adj):
        adj_used = refine_adjacency_from_embeddings(adj, x, self.dynamic_graph_k, self.dynamic_graph_blend) if self.dynamic_graph_refine else adj
        g = F.relu(self.graph_proj(self.gcn_encoder(x, adj_used)))
        tokens = self.token_proj(x).view(x.shape[0], self.token_count, -1)
        t = self.transformer(tokens).mean(dim=1)
        t = F.relu(self.trans_proj(t))
        gate = self.fusion_gate(torch.cat([g, t], dim=1))
        fused = self.norm(gate * g + (1.0 - gate) * t)
        fused = self.dropout(fused)
        self.last_features = fused
        logits = self.prototype_classifier(fused)
        return {"logits": logits, "embedding": fused, "gate": gate, "adjacency": adj_used}

    def contrastive_loss(self, nodes, labels):
        if self.last_features is None or labels.numel() < 2:
            return torch.tensor(0.0, device=labels.device)
        z = F.normalize(self.last_features[nodes], dim=1)
        sim = z @ z.t() / self.contrastive_temperature
        eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        same = labels[:, None].eq(labels[None, :]) & ~eye
        sim = sim.masked_fill(eye, -1e9)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        valid = same.any(dim=1)
        if not valid.any():
            return torch.tensor(0.0, device=labels.device)
        per_node = -(log_prob * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp_min(1.0)
        return per_node[valid].mean()
