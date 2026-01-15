import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from dataclasses import dataclass
from dataclasses import make_dataclass
import sys  # <-- add this
import pickle
from captum.attr import Occlusion
from Bio import SeqIO
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict

# MODEL
class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        pos = torch.arange(L, device=x.device).clamp_max(self.pe.num_embeddings - 1).unsqueeze(0)
        return x + self.pe(pos)
    
class ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) // 2 * dilation
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=padding, dilation=dilation, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)                  # [B,D,L]
        y = F.gelu(y).transpose(1, 2)     # [B,L,D]
        y = self.ln(y).transpose(1, 2)
        y = self.dropout(y)
        out = x + y
        return out * mask_1d.unsqueeze(1).to(out.dtype)

class GatedPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(6.0)))
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.proj(x).squeeze(-1)            # [B,L]
        B, L = scores.size()
        pos = torch.arange(L, device=x.device).float().unsqueeze(0).expand(B, -1)
        lens = (mask.sum(1) if mask is not None else torch.full((B,), L, device=x.device)).clamp(min=1).float().unsqueeze(1)
        centers = (lens - 1) / 2
        dist2 = (pos - centers).pow(2)
        sigma = torch.exp(self.log_sigma) + 1e-6
        scores = scores + self.alpha * (- dist2 / (2 * sigma * sigma))
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores.float(), dim=1).to(x.dtype)
        return torch.einsum('bl,bld->bd', attn, x)
    
class ProtMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(p),
            nn.Linear(hidden, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.net(x))
    
class RNATower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab = {ch: i for i, ch in enumerate(cfg.RNA_VOCAB)}
        self.pad_id = len(cfg.RNA_VOCAB)
        V, D = len(cfg.RNA_VOCAB), cfg.D_MODEL   # D = 256
        self.embed = nn.Embedding(V + 1, D, padding_idx=self.pad_id) # The nucleutide embedding, the <PAD> vector is always zero.
        self.pos = PositionalEncoding(D, cfg.RNA_MAX_LEN)

        self.use_struct = bool(getattr(cfg, "RNA_USE_STRUCT", False))
        if self.use_struct:

            self.struct_proj = nn.Linear(getattr(cfg, "RNA_STRUCT_DIM", 5), D, bias=True)
            self.struct_ln = nn.LayerNorm(D)
            self.struct_drop = nn.Dropout(cfg.RNA_DROPOUT)
            self.struct_scale = nn.Parameter(torch.tensor(1.0)) #is a learnable scalar that controls how much structural signal gets injected.
        else:
            self.struct_proj = None

        self.conv1 = ConvBlock(D, 5, 1, cfg.RNA_DROPOUT)
        self.conv2 = ConvBlock(D, 9, 2, cfg.RNA_DROPOUT)
        self.conv3 = ConvBlock(D, 13, 4, cfg.RNA_DROPOUT)
        self.k9 = nn.Conv1d(in_channels=D, out_channels=D, kernel_size=9, padding=4, bias=False) 
        self.k9_gamma = nn.Parameter(torch.tensor(0.5))

        if cfg.RNA_USE_TRANSFORMER:
            el = nn.TransformerEncoderLayer(d_model=D, nhead=cfg.RNA_NHEAD, dim_feedforward=D*4,
                                            dropout=cfg.RNA_DROPOUT, batch_first=True)
            self.tf = nn.TransformerEncoder(el, num_layers=cfg.RNA_TRANSFORMER_LAYERS)
        else:
            self.tf = None

        self.pool = GatedPooling(D)
        self.out_norm = nn.LayerNorm(D)

    def forward_from_emb(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                # We already have embeddings (x), so we skip self.embed
                # We receive the mask directly
                
                x = self.pos(x)
                x = x * mask.unsqueeze(-1).to(x.dtype)  # Padding tokens → set to zero.

                xc = x.transpose(1, 2)                              # [B,D,L]
                xc = self.conv1(xc, mask)
                xc = self.conv2(xc, mask)
                xc = self.conv3(xc, mask)
                k9 = F.gelu(self.k9(xc))
                k9 = k9 * mask.unsqueeze(1).to(k9.dtype)
                xc = xc + self.k9_gamma * k9
                x = xc.transpose(1, 2)

                if self.tf is not None:
                    x = self.tf(x, src_key_padding_mask=~mask)
                h = self.pool(x, mask)
                return self.out_norm(h)

    def forward(self, tokens: torch.Tensor, struct: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = (tokens != self.pad_id)                      # [B,L]
        x = self.embed(tokens)                              # [B,L,D]
        x = self.pos(x)
        x = x * mask.unsqueeze(-1).to(x.dtype)  # Padding tokens → set to zero.

        if self.use_struct and struct is not None and self.struct_proj is not None:
            s = self.struct_proj(struct.to(x.dtype))
            s = self.struct_ln(F.gelu(s))
            s = self.struct_drop(s)
            x = x + self.struct_scale * s
            x = x * mask.unsqueeze(-1).to(x.dtype)    # again Padding tokens → set to zero.

        xc = x.transpose(1, 2)                              # [B,D,L]
        xc = self.conv1(xc, mask)
        xc = self.conv2(xc, mask)
        xc = self.conv3(xc, mask)
        k9 = F.gelu(self.k9(xc))
        k9 = k9 * mask.unsqueeze(1).to(k9.dtype)
        xc = xc + self.k9_gamma * k9
        x = xc.transpose(1, 2)

        if self.tf is not None:
            x = self.tf(x, src_key_padding_mask=~mask)
        h = self.pool(x, mask)
        return self.out_norm(h)
    
class GatedBilinearLowRankCosine(nn.Module):

    def __init__(self, dim: int, rank: int, gate_strength: float = 0.5):
        super().__init__()
        self.U = nn.Linear(dim, rank, bias=False)   # protein -> rank
        self.V = nn.Linear(dim, rank, bias=False)   # RNA -> rank
        self.G = nn.Linear(dim, rank, bias=True)    # protein -> gate
        self.bias = nn.Parameter(torch.zeros(1))
        self.gate_strength = gate_strength
    def forward(self, e_p: torch.Tensor, e_r: torch.Tensor) -> torch.Tensor:
        up = F.normalize(self.U(e_p), dim=1)        # [Bp,R]
        vr = F.normalize(self.V(e_r), dim=1)        # [Br,R]
        g  = torch.tanh(self.G(e_p))                # [Bp,R]
        upg = up * (1.0 + self.gate_strength * g)
        return upg @ vr.t() + self.bias   # final score matrix [Bp, Br]. B=batch    

class TwoTowerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rna = RNATower(cfg)
        self.prot_proj = ProtMLP(cfg.PROT_EMB_DIM, cfg.PROT_MLP_HIDDEN, cfg.D_MODEL, cfg.PROT_DROPOUT)
        self.score = GatedBilinearLowRankCosine(cfg.D_MODEL, cfg.RANK, cfg.GATE_STRENGTH)

    def encode_rna(self, rna_tokens: torch.Tensor, rna_struct: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.rna(rna_tokens, rna_struct)

    def project_prot(self, prot_vecs: torch.Tensor) -> torch.Tensor:
        return self.prot_proj(prot_vecs)

    def forward_scores_vecs(
        self,
        prot_vecs: torch.Tensor,
        rna_tokens: torch.Tensor,
        rna_struct: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        e_p = self.project_prot(prot_vecs)
        e_r = self.encode_rna(rna_tokens, rna_struct)
        return self.score(e_p, e_r)


def read_fasta(fasta_file):
    sequences = []
    protein_ids = []
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        # Each seq_record object contains id, description, and sequence
        sequences.append(str(seq_record.seq)) # Extract the sequence as a string
        protein_ids.append(str(seq_record.id))
    print(f"Extracted {len(sequences)} sequences.")

    return sequences, protein_ids



# =============================
# Tokenization & batching
# =============================
vocab = 'ACGT'
PAD_ID = len(vocab) 

NA_TO_IDX = {ch: i for i, ch in enumerate(vocab)} 
NA_TO_IDX['U'] = 3

def tokenize_rna_batch(seqs: List[str]) -> torch.Tensor:
    lens = [len(s) for s in seqs]
    max_len = max(lens) if lens else 0
    print("max rna len", max_len)
    out = torch.full((len(seqs), max_len), fill_value=PAD_ID, dtype=torch.long)  # fill with PAD
    for i, s in enumerate(seqs):
        ids = [NA_TO_IDX.get(ch, 0) for ch in s]  # unknowns → 'A' (0), fine for RNAcompete
        if ids:
            out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out



def forward_fn(emb_inputs, mask, eval_model, protein_embedding_proj):
    """
    Captum forward function.
    """
    eval_model.eval()
    
    if mask.shape[0] == 1 and emb_inputs.shape[0] > 1:
        mask = mask.repeat(emb_inputs.shape[0], 1)
        
    encode_rna = eval_model.rna.forward_from_emb(emb_inputs, mask)
    score = eval_model.score(protein_embedding_proj, encode_rna) # Shape [1, B]
    
    return score.t() # Transpose to [B, 1]
def load_dynamic_config(cfg_dict: dict):
    normalized_fields = []
    
    # Mapping for name normalization
    name_mapping = {
        'TRAIN_MATRIX': 'TRAIN_MATRIX_PATH',
        'TEST_MATRIX':  'TEST_MATRIX_PATH',
        'NA_VOCAB': 'RNA_VOCAB',
        'NA_MAX_LEN': 'RNA_MAX_LEN',
        'NA_DROPOUT': 'RNA_DROPOUT',
        'NA_USE_TRANSFORMER': 'RNA_USE_TRANSFORMER',
        'NA_NHEAD': 'RNA_NHEAD',
        'NA_TRANSFORMER_LAYERS': 'RNA_TRANSFORMER_LAYERS'
    }

    for key, value in cfg_dict.items():
        target_key = name_mapping.get(key, key)
        
        # De-duplication check
        if any(f[0] == target_key for f in normalized_fields):
            continue 

        inferred_type = type(value)

        # FIX: Check if the value is a list (or other mutable)
        if isinstance(value, list):
            # We use a lambda to return a copy of the list as the default
            normalized_fields.append((
                target_key, 
                inferred_type, 
                field(default_factory=lambda v=value: list(v))
            ))
        else:
            # Standard immutable types (str, int, float, bool) are fine as-is
            normalized_fields.append((target_key, inferred_type, value))

    DynamicConfig = make_dataclass("DynamicConfig", normalized_fields)
    return DynamicConfig()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="run interpretability")
    parser.add_argument("protein_name", help="Name of the protein")
    parser.add_argument("protein_embedding", help="Path to input protein esm2 embedding")
    parser.add_argument("na_fasta", help="Path to input RNA/DNA FASTA file")
    parser.add_argument("nucleic_acid_type", help="RNA/DNA")
    parser.add_argument("run_id", type=str, help="run ID for saving/loading files")
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    args = parser.parse_args()

    run_id = args.run_id
    PROTEIN = args.protein_name

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(THIS_DIR, "../../output", run_id, "interpretability") 
    model_dir = os.path.join(THIS_DIR, "../../models/NucProNet")

    os.makedirs(output_dir, exist_ok=True)

    nuc_type = args.nucleic_acid_type.upper()
    
    if nuc_type == "RNA":
        vocab = 'ACGU'
        model_path = os.path.join(model_dir, "NucProNet_RNA.pt")
    elif nuc_type == "DNA":
        vocab = 'ACGT'
        model_path = os.path.join(model_dir, "NucProNet_DNA.pt")
    else:
        print("error must specified nucleic_acid_type DNA or RNA")
        sys.exit(1)
    if args.model_path:
        model_path = args.model_path

    na_fasta = args.na_fasta
    na_sequences, na_ids = read_fasta(na_fasta)
    print(f"Loaded {len(na_sequences)}")

    protein_embedding_path = args.protein_embedding
    protein_embedding = torch.tensor(np.load(protein_embedding_path).astype(np.float32)).to(device)
    if protein_embedding.dim() == 1:
        protein_embedding = protein_embedding.unsqueeze(0)
    print(f"Loaded protein embedding of shape: {protein_embedding.shape}")

    # Load the MODEL
    payload = torch.load(model_path, map_location=device)
    cfg = load_dynamic_config(payload["cfg"])
    model = TwoTowerModel(cfg).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    print("Loaded pre-trained model.")

    # --- Pre-compute protein vector ---
    with torch.no_grad():
        protein_embedding_proj = model.project_prot(protein_embedding) # [1, D_model]


    # This dictionary will hold all our results
    all_attributions = {} 


    occlusion_forward_fn = lambda emb_inputs, mask: forward_fn(emb_inputs, mask, model, protein_embedding_proj)
    occlusion = Occlusion(occlusion_forward_fn)
    

    # ========================================================
    # Loop through all probes and run occlusion
    # ========================================================
   
    for i, seq in enumerate(tqdm(na_sequences, desc="Analyzing Top Probes")):
        
        try:
            tokens = tokenize_rna_batch([seq]).to(device)
            mask = (tokens != PAD_ID).to(device)
            emb = model.rna.embed(tokens)
            base_emb = torch.zeros_like(emb)

            # Choose occlusion window size (k-mer). Use 1 for single-nucleotide occlusion.
            win_k = 3

            # embedding dimension
            emb_dim = emb.size(-1)   # D

            # sliding_window_shapes must match emb.shape (including batch dim)
            # emb shape = [1, L, D] -> sliding window shape = (1, k, D)
            sliding_window_shapes = (win_k, emb_dim)

            # Optionally set strides (default stride=1). For non-overlapping windows use (1, win_k, emb_dim)
            strides = (1, emb_dim)  # here we occlude 1 pos at a time (overlap), and all embedding dims

            # Perform occlusion
            attributions = occlusion.attribute(
                inputs=emb,                      # float tensor [1, L, D]
                baselines=base_emb,              # same shape, zeros
                additional_forward_args=(mask,), # mask is required by forward_fn
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
                target=None                      # model returns scalar per sequence, so None
            )

            # attributions has same shape as inputs: [1, L, D]
            # get per-position importance by summing/ norm across embedding dim
            attributions_per_token = attributions.sum(dim=-1).squeeze(0)  # shape [L]
            scores_np = attributions_per_token.cpu().detach().numpy()

            
            # --- NEW: Save the scores to the dictionary ---
            seq_id = na_ids[i] # Get the corresponding sequence ID
            all_attributions[seq_id] = {'attributions':scores_np, 'seq':seq}
            
        except Exception as e:
            print(f"Error processing sequence {seq[:10]}... (ID: {na_ids[i]}): {e}")
            pass # Continue to the next sequence

    # ========================================================
    # Save all attributions to a file
    # ========================================================
    output_filename = f'{PROTEIN}_attributions_scores.pkl'
    dest = os.path.join(output_dir, output_filename)
    
    print(f"\nAnalysis complete. Saving {len(all_attributions)} attribution vectors to {dest}...")
    
    with open(dest, 'wb') as f:
        pickle.dump(all_attributions, f)