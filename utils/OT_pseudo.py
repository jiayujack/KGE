#!/usr/bin/env python3
"""
Script for pseudo-label generation for protein–molecule interactions using Sinkhorn OT.

Steps:
1. Load a pre-trained PyTorch scoring model.
2. Load protein and molecule embeddings from the given directory.
3. Load the PSI knowledge graph (parquet) to get known protein–molecule pairs.
4. Compute Morgan fingerprints (radius=2) for molecules (using RDKit) and Tanimoto similarity.
5. Build a cost matrix: C[p,m] = -score(p,m) - λ * max_sim(m, known_actives(p)).
6. Solve entropic-regularized optimal transport (Sinkhorn) using POT.
7. Save top-k pseudo-labeled molecule matches per protein to 'pseudo_labels.txt'.

Defaults: λ=0.5, ε=0.1, top_k=50.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import ot

def load_embeddings(emb_dir, ids_filter=None):
    """
    Load embeddings from a directory. Files should be named by ID (without extension).
    Supports .npy or .pt files containing numpy arrays or torch tensors.
    If ids_filter is provided, only load embeddings for those IDs.
    Returns a dict: id -> numpy array.
    """
    emb_dict = {}
    for fname in os.listdir(emb_dir):
        if not (fname.endswith('.npy') or fname.endswith('.pt')):
            continue
        key = os.path.splitext(fname)[0]
        if ids_filter is not None and key not in ids_filter:
            continue
        path = os.path.join(emb_dir, fname)
        try:
            if fname.endswith('.npy'):
                emb = np.load(path)
            else:
                emb = torch.load(path)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
            emb_dict[key] = emb
        except Exception as e:
            print(f"Warning: Failed to load embedding {fname}: {e}")
    return emb_dict

def main():
    parser = argparse.ArgumentParser(description="Pseudo-label generation using Sinkhorn OT")
    parser.add_argument('--model', type=str, default='score_model.pt', help="Trained scoring model (.pt)")
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings', help="Directory with embeddings (.npy or .pt)")
    parser.add_argument('--kg', type=str, default='PSI_kg_full.parquet', help="PSI knowledge graph (parquet)")
    parser.add_argument('--lambda_', type=float, default=0.5, help="Lambda coefficient for similarity term")
    parser.add_argument('--epsilon', type=float, default=0.1, help="Entropy regularization (epsilon) for Sinkhorn")
    parser.add_argument('--top_k', type=int, default=50, help="Top-k pseudo-labeled molecules per protein")
    args = parser.parse_args()

    # Load scoring model
    print("Loading scoring model...")
    model = torch.load(args.model)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load PSI knowledge graph
    print("Loading knowledge graph...")
    kg = pd.read_parquet(args.kg)
    # Identify protein and molecule columns (attempt to detect by name)
    prot_col = None
    for col in kg.columns:
        if 'prot' in col.lower():
            prot_col = col
            break
    if prot_col is None:
        prot_col = kg.columns[0]
    mol_col = next((col for col in kg.columns if col != prot_col), None)
    if mol_col is None:
        raise ValueError("Could not identify molecule column in KG.")
    # Group known actives by protein
    known_actives = {}
    for _, row in kg.iterrows():
        p = row[prot_col]
        m = row[mol_col]
        known_actives.setdefault(p, []).append(m)
    proteins_list = set(known_actives.keys())
    print(f"Found {len(proteins_list)} proteins in KG with known actives.")

    # Load embeddings: proteins of interest and all molecules
    print("Loading embeddings...")
    protein_embs = load_embeddings(args.embeddings_dir, ids_filter=proteins_list)
    # Check for missing proteins
    missing = proteins_list - set(protein_embs.keys())
    if missing:
        print(f"Warning: {len(missing)} proteins from KG have no embeddings. They will be skipped: {missing}")
        for pid in missing:
            known_actives.pop(pid, None)
        proteins_list = set(protein_embs.keys())
    # Load all embeddings
    all_embs = load_embeddings(args.embeddings_dir)
    # Separate molecules (exclude protein IDs)
    molecule_ids = [mid for mid in all_embs.keys() if mid not in proteins_list]
    print(f"Loaded {len(protein_embs)} protein embeddings, {len(molecule_ids)} molecule embeddings.")

    # Attempt to get SMILES for molecules from KG (if present)
    smiles_col = next((col for col in kg.columns if 'smile' in col.lower()), None)
    smiles_map = {}
    if smiles_col:
        smiles_map = dict(zip(kg[mol_col], kg[smiles_col]))
    else:
        print("Warning: No SMILES column found in KG. Tanimoto similarities will be zero.")

    # Pre-compute Morgan fingerprints (radius=2) for candidate molecules with known SMILES
    print("Computing fingerprints for candidate molecules...")
    cand_fps = {}
    for mid in molecule_ids:
        smi = smiles_map.get(mid, None)
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                cand_fps[mid] = fp

    # Build cost matrix (proteins x molecules)
    proteins = sorted(protein_embs.keys())
    molecules = sorted(molecule_ids)
    n_prot = len(proteins)
    n_mol = len(molecules)
    print(f"Building cost matrix of size ({n_prot} x {n_mol})...")
    cost_matrix = np.zeros((n_prot, n_mol), dtype=np.float64)

    with torch.no_grad():
        for i, prot in enumerate(proteins):
            prot_emb = torch.tensor(protein_embs[prot], dtype=torch.float32, device=device)
            # Prepare fingerprints of known actives for this protein
            known_ids = known_actives.get(prot, [])
            known_smiles = [smiles_map[mid] for mid in known_ids if smiles_map.get(mid)]
            known_fps = []
            for smi in known_smiles:
                kmol = Chem.MolFromSmiles(smi)
                if kmol:
                    known_fps.append(AllChem.GetMorganFingerprintAsBitVect(kmol, radius=2, nBits=2048))
            # Loop over candidate molecules
            for j, mol in enumerate(molecules):
                mol_emb = torch.tensor(all_embs[mol], dtype=torch.float32, device=device)
                # Compute score from the model
                try:
                    score_tensor = model(prot_emb.unsqueeze(0), mol_emb.unsqueeze(0))
                except TypeError:
                    # If model expects concatenated input
                    inp = torch.cat([prot_emb, mol_emb]).unsqueeze(0)
                    score_tensor = model(inp)
                score_val = score_tensor.item() if isinstance(score_tensor, torch.Tensor) else float(score_tensor)

                # Compute max Tanimoto similarity to known actives
                sim = 0.0
                if known_fps:
                    if mol in cand_fps:
                        cand_fp = cand_fps[mol]
                        sim = max(DataStructs.TanimotoSimilarity(cand_fp, kfp) for kfp in known_fps)
                    else:
                        # If candidate SMILES available, compute on-the-fly
                        smi_cand = smiles_map.get(mol, None)
                        if smi_cand:
                            cand_mol = Chem.MolFromSmiles(smi_cand)
                            if cand_mol:
                                cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, radius=2, nBits=2048)
                                sim = max(DataStructs.TanimotoSimilarity(cand_fp, kfp) for kfp in known_fps)
                # Fill cost matrix
                cost_matrix[i, j] = -score_val - args.lambda_ * sim

    # Compute Sinkhorn transport plan (entropic OT)
    print("Running Sinkhorn OT...")
    a = np.ones(n_prot) / n_prot
    b = np.ones(n_mol) / n_mol
    transport_plan = ot.sinkhorn(a, b, cost_matrix, args.epsilon)  # ε=regularization:contentReference[oaicite:2]{index=2}

    # Extract top-k pseudo-labels
    print(f"Selecting top-{args.top_k} pseudo-labeled pairs for each protein...")
    with open('pseudo_labels.txt', 'w') as fout:
        for i, prot in enumerate(proteins):
            row = transport_plan[i]
            top_idx = np.argsort(row)[::-1][:args.top_k]
            for idx in top_idx:
                fout.write(f"{prot}\t{molecules[idx]}\n")

    print("Finished. Pseudo-labels saved to pseudo_labels.txt.")

if __name__ == "__main__":
    main()
