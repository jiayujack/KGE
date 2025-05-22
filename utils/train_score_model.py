import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Dataset for molecule-protein interaction pairs
class MolProtInteractionDataset(Dataset):
    def __init__(self, interaction_file, embedding_dir):
        self.pairs = []
        with open(interaction_file, "r") as f:
            for line in f:
                prot, _, mol = line.strip().split("\t")
                prot = prot.replace(":", "_")
                mol = mol.replace(":", "_")
                mol_path = os.path.join(embedding_dir, f"{mol}.npy")
                prot_path = os.path.join(embedding_dir, f"{prot}.npy")
                if os.path.exists(mol_path) and os.path.exists(prot_path):
                    self.pairs.append((mol_path, prot_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mol_path, prot_path = self.pairs[idx]
        mol_emb = np.load(mol_path)
        prot_emb = np.load(prot_path)
        return torch.tensor(mol_emb, dtype=torch.float32), torch.tensor(prot_emb, dtype=torch.float32)

# Scoring model S(x, y)
class ScoreModel(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, mol_emb, prot_emb):
        mol_emb = nn.functional.normalize(mol_emb, dim=-1)
        prot_emb = nn.functional.normalize(prot_emb, dim=-1)
        concat = torch.cat([mol_emb, prot_emb], dim=-1)
        return self.mlp(concat).squeeze(-1)

# Sinkhorn function
def sinkhorn(log_alpha, n_iter=50, eps=1e-8):
    N = log_alpha.size(0)
    log_r = torch.full((N,), -np.log(N), device=log_alpha.device)
    log_c = torch.full((N,), -np.log(N), device=log_alpha.device)
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True) + log_r.view(-1, 1)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True) + log_c.view(1, -1)
    return torch.exp(log_alpha)

# KL divergence loss between transport matrices
def transport_loss(scores):
    N = scores.size(0)
    C_pred = 1.0 - scores.unsqueeze(1) + torch.zeros(1e-8, device=scores.device)
    log_alpha = -C_pred
    T_pred = sinkhorn(log_alpha)
    C_gt = torch.ones_like(C_pred)
    C_gt.fill_diagonal_(0.0)
    log_alpha_gt = -C_gt
    T_gt = sinkhorn(log_alpha_gt)
    loss = torch.sum(T_pred * (torch.log(T_pred + 1e-8) - torch.log(T_gt + 1e-8)))
    return loss

# Training loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MolProtInteractionDataset("train.txt", "embeddings")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = ScoreModel(emb_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for mol_embs, prot_embs in dataloader:
            mol_embs = mol_embs.to(device)
            prot_embs = prot_embs.to(device)
            scores = model(mol_embs.unsqueeze(1).repeat(1, mol_embs.size(0), 1).view(-1, 256),
                           prot_embs.unsqueeze(0).repeat(mol_embs.size(0), 1, 1).view(-1, 256)).view(mol_embs.size(0), -1)
            loss = transport_loss(scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "score_model.pt")

if __name__ == "__main__":
    train()
