import os, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List
import sys
from typing import List, Tuple, Optional, Dict
import time
import pickle
from Bio import SeqIO
import argparse
import pandas as pd



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUCLEIC_ACIDS = 'ACGT'
PAD_ID = len(NUCLEIC_ACIDS)               
NA_TO_IDX = {ch: i for i, ch in enumerate(NUCLEIC_ACIDS)}  # real tokens are 0..3
NA_TO_IDX['U'] = 3 
@dataclass
class Config:

    # File paths 
    TRAIN_MATRIX = None     # (N_NA, N_rbp)
    TRAIN_NA_SEQS: str = ''       

    # --- EMA of model parameters ---
    EMA_USE: bool = False          # turn EMA on/off
    EMA_EVAL: bool = False         # use EMA weights for evaluation/checkpoints

    # --- NEW: NA structure features (PHIME: P,H,I,M,E) ---
    NA_USE_STRUCT: bool = False                 # flip to True after generating NPZs

    # Random seeds
    SEED: int = 1

    # Model dims
    D_MODEL: int = 256
    NA_VOCAB: str = ''      # T will be mapped to U  
    RANK: int = 512
    GATE_STRENGTH: float = 0.5
    # NA tower
    NA_USE_TRANSFORMER: bool = True
    NA_TRANSFORMER_LAYERS: int = 2
    NA_NHEAD: int = 4
    NA_DROPOUT: float = 0.3
    NA_MAX_LEN: int = 64        # safety cap for positional embeddings

    # Training
    BATCH_RBPS: int = 8          # proteins per batch 8
    BATCH_NAS: int = 2048       # NAs per batch 2048
    EPOCHS: int = 1  #50
    STEPS_PER_EPOCH: int = 5000 #300   # tune to time budget  300
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-2
    HUBER_DELTA: float = 1.0
    CORR_WEIGHT: float = 0.8     # loss = (1-CORR_WEIGHT)*Huber + CORR_WEIGHT*(1-Pearson)
    MIXED_PRECISION: bool = False

    # Validation setup
    USE_KMER_DISJOINT_NA: bool = False
    VAL_RBPS_COUNT: int = 0
    VAL_RBPS_SEED: int = 176
    VAL_RBPS_INDICES: Optional[List[int]] = None  # can be original indices; will auto-map if dedup applied


    # Target preprocessing
    LOG1P: bool = False
    CLIP_PCTL: float = 99.5

    # Evaluation / caching
    EVAL_ZSCORE_TARGETS: bool = False
    CACHE_DIR: str = 'cache'
    SAVE_EVERY: int = 2

    # ProtT5
    USE_PROTT5: bool = False

    # ESM-2 (HF names supported; short names via map below)
    USE_ESM2: bool = True
    ESM2_MODEL_ID: str = "facebook/esm2_t48_15B_UR50D"
    #ESM_EMB_DIM: int = 1280            # info only
    ESM_EMB_BATCH: int = 8
    ESM_EMB_CACHE_TRAIN: str = ''

    # Fusion hygiene / regularization
    PROT_SRC_ZSCORE: bool = True
    PROT_SWAPDROP_P: float = 0.0

    # Prot MLP
    PROT_MLP_HIDDEN: int = 256
    PROT_DROPOUT: float = 0.3
    PROT_MLP_WD: float = 3e-4


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
    
class NATower(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.vocab = {ch: i for i, ch in enumerate(cfg.NA_VOCAB)}
        self.pad_id = PAD_ID
        V, D = len(cfg.NA_VOCAB), cfg.D_MODEL   # D = 256
        self.embed = nn.Embedding(V + 1, D, padding_idx=self.pad_id) # The nucleutide embedding, the <PAD> vector is always zero.
        self.pos = PositionalEncoding(D, cfg.NA_MAX_LEN)

        self.use_struct = bool(getattr(cfg, "NA_USE_STRUCT", False))
        if self.use_struct:

            self.struct_proj = nn.Linear(getattr(cfg, "NA_STRUCT_DIM", 5), D, bias=True)
            self.struct_ln = nn.LayerNorm(D)
            self.struct_drop = nn.Dropout(cfg.NA_DROPOUT)
            self.struct_scale = nn.Parameter(torch.tensor(1.0)) #is a leaNAble scalar that controls how much structural signal gets injected.
        else:
            self.struct_proj = None

        self.conv1 = ConvBlock(D, 5, 1, cfg.NA_DROPOUT)
        self.conv2 = ConvBlock(D, 9, 2, cfg.NA_DROPOUT)
        self.conv3 = ConvBlock(D, 13, 4, cfg.NA_DROPOUT)
        self.k9 = nn.Conv1d(in_channels=D, out_channels=D, kernel_size=9, padding=4, bias=False) 
        self.k9_gamma = nn.Parameter(torch.tensor(0.5))

        if cfg.NA_USE_TRANSFORMER:
            el = nn.TransformerEncoderLayer(d_model=D, nhead=cfg.NA_NHEAD, dim_feedforward=D*4,
                                            dropout=cfg.NA_DROPOUT, batch_first=True)
            self.tf = nn.TransformerEncoder(el, num_layers=cfg.NA_TRANSFORMER_LAYERS)
        else:
            self.tf = None

        self.pool = GatedPooling(D)
        self.out_norm = nn.LayerNorm(D)

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
        self.V = nn.Linear(dim, rank, bias=False)   # NA -> rank
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
    def __init__(self, cfg: Config):
        super().__init__()
        self.NA = NATower(cfg)
        self.prot_proj = ProtMLP(cfg.PROT_EMB_DIM, cfg.PROT_MLP_HIDDEN, cfg.D_MODEL, cfg.PROT_DROPOUT)
        self.score = GatedBilinearLowRankCosine(cfg.D_MODEL, cfg.RANK, cfg.GATE_STRENGTH)

    def encode_NA(self, NA_tokens: torch.Tensor, NA_struct: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.NA(NA_tokens, NA_struct)

    def project_prot(self, prot_vecs: torch.Tensor) -> torch.Tensor:
        return self.prot_proj(prot_vecs)

    def forward_scores_vecs(
        self,
        prot_vecs: torch.Tensor,
        NA_tokens: torch.Tensor,
        NA_struct: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        e_p = self.project_prot(prot_vecs)
        e_r = self.encode_NA(NA_tokens, NA_struct)
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


class TrainData:

    """Holds full training tensors and index splits; also saves μ,σ for z-scoring."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        print("\n[Load] Reading training files ...")

        NA_seqs = cfg.TRAIN_NA_SEQS
        Y = cfg.TRAIN_MATRIX.astype(np.float32)   # (N_NA, N_rbp)
        total_NA, total_proteins = Y.shape
        assert Y.shape[0] == len(NA_seqs), "Rows of matrix must equal number of NA sequences"
        self.NA_train_seqs = NA_seqs
        self.Y = Y         
        self.na_idx = np.arange(total_NA, dtype=np.int64)
        self.protein_idx = np.arange(total_proteins, dtype=np.int64)

# =============================
# Losses & metrics
# =============================
class PearsonLoss(nn.Module):
    """1 - Pearson, averaged across protein rows in the mini-batch."""
    def __init__(self, eps: float = 1e-8): super().__init__(); self.eps = eps
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        cov = (pred * target).sum(dim=1)
        denom = torch.sqrt((pred.pow(2).sum(dim=1) + self.eps) * (target.pow(2).sum(dim=1) + self.eps))
        r = cov / denom
        return (1.0 - r).mean()

def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)

def pearson_numpy(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a*a).sum() * (b*b).sum()) + 1e-12
    return float((a*b).sum() / denom)

# =============================
# Tokenization & batching
# =============================

def tokenize_NA_batch(seqs: List[str]) -> torch.Tensor:
    lens = [len(s) for s in seqs]
    max_len = max(lens) if lens else 0
    out = torch.full((len(seqs), max_len), fill_value=PAD_ID, dtype=torch.long)  # fill with PAD
    for i, s in enumerate(seqs):
        ids = [NA_TO_IDX.get(ch, 0) for ch in s]  # unknowns → 'A' (0), fine for NAcompete
        if ids:
            out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out

class BatchSampler:
    """Yields (rbp_idx_list, NA_idx_list) for each training step."""
    def __init__(self, data: TrainData, cfg: Config):
        self.data, self.cfg = data, cfg
        self.rng = np.random.default_rng(cfg.SEED)
        
    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        rbps = self.rng.choice(self.data.protein_idx, size=self.cfg.BATCH_RBPS, replace=True)
        NAs = self.rng.choice(self.data.na_idx,   size=self.cfg.BATCH_NAS, replace=False)
        return rbps.astype(np.int64), NAs.astype(np.int64)

def load_esm2_embeddings(cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        E = np.load(cache_path)
        return E.astype(np.float32)
    else:
        print("path no right")


class Trainer:
    def __init__(self, data: TrainData, cfg: Config):
        self.data, self.cfg = data, cfg

        # load embedding !!!
        esm_np = load_esm2_embeddings(cfg.ESM_EMB_CACHE_TRAIN)
        self.esm_all = torch.tensor(esm_np, dtype=torch.float32, device=device)  # [N_rbp, d]
        total_dim = self.esm_all.shape[1]
        cfg.PROT_EMB_DIM = total_dim

        # Build model !!!
        self.model = TwoTowerModel(cfg).to(device)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Trainable parameters:", count_parameters(self.model))

        # optimezer and loss !!!
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        self.pearson_loss = PearsonLoss()
        self.best_median = -1.0
        self.checkpoint_path = os.path.join(cfg.CACHE_DIR, 'best_model.pt')


    def _prep_prot(self, rbp_idx: torch.Tensor) -> torch.Tensor:
        esm = None
        if self.esm_all is not None:
            esm = self.esm_all[rbp_idx]
        return esm

    def sample_batch(self, sampler: BatchSampler):
        rbp_idx_np, NA_idx = sampler.sample()
        rbp_idx = torch.tensor(rbp_idx_np, device=device)
        prot_vecs = self._prep_prot(rbp_idx)                       
        NA_tokens = tokenize_NA_batch([self.data.NA_train_seqs[i] for i in NA_idx]).to(device)
        Y = torch.tensor(self.data.Y[np.ix_(NA_idx, rbp_idx_np)], device=device).t()# [Bp,Br]
        return prot_vecs, NA_tokens, Y 
            
    def train(self):
        sampler = BatchSampler(self.data, self.cfg)
        for epoch in range(1, self.cfg.EPOCHS + 1):
            self.model.train() 
            losses = []
            t0 = time.time()
            for step in range(1, self.cfg.STEPS_PER_EPOCH + 1):
                prot_vecs, NA_tokens, Yz = self.sample_batch(sampler)
                self.opt.zero_grad(set_to_none=True)


                # Forward pass
                S = self.model.forward_scores_vecs(prot_vecs, NA_tokens, NA_struct=None)  # [Bp, Br]
                loss = (1 - self.cfg.CORR_WEIGHT) * huber_loss(S, Yz, delta=self.cfg.HUBER_DELTA) \
                    + self.cfg.CORR_WEIGHT * self.pearson_loss(S, Yz)
                
                # Backward + update
                loss.backward()
                self.opt.step()

                losses.append(loss.item())
                if step % 50 == 0:
                    print(f"Epoch {epoch} Step {step}/{self.cfg.STEPS_PER_EPOCH} | loss={np.mean(losses[-50:]):.4f}")

        return self.model


def main():

    # get input from the user
    parser = argparse.ArgumentParser(description="Train NucProNet model on DNA/NA sequences.")
    parser.add_argument("nucleic_acid_type", help="RNA or DNA")
    parser.add_argument("model_name", help="Name of the model")
    parser.add_argument("--protein_ids", help="line-separated file of protein IDs for training")
    parser.add_argument("--embedding_path", help="Path to the embedding file")
    args = parser.parse_args()

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(THIS_DIR, "../../models/NucProNet")
    intensities_dir = os.path.join(THIS_DIR, "../../data/intensities")
    embeddings_dir = os.path.join(THIS_DIR, "../../data/embeddings")

    nuc_type = args.nucleic_acid_type
    embedding_path = args.embedding_path  # #_protein X embedding_dim (esm2_t48_15B_UR50D)

    cfg = Config()
    if args.protein_ids is None:
        protein_ids = []
    else:
        protein_ids = []
        with open(args.protein_ids, "r") as f:
            for line in f:
                protein_id = line.strip()
                protein_ids.append(protein_id)

    if nuc_type.upper() == "RNA":
        cfg.NA_VOCAB = 'ACGU'
        intensities_file = os.path.join(intensities_dir, "RNA_norm_data_420.csv")
        data = pd.read_csv(intensities_file)
        if 'NA_Seq' not in data.columns:
            if 'RNA_Seq' in data.columns:
                data = data.rename(columns={'RNA_Seq': 'NA_Seq'})
            else:
                print("Neither 'NA_Seq' nor 'RNA_Seq' found.")
                sys.exit(1)

        if protein_ids:
            # Filter columns based on provided protein IDs
            cols_to_keep = ['NA_Seq', 'Probe_Set'] + protein_ids
            data = data[cols_to_keep]
            cfg.ESM_EMB_CACHE_TRAIN = embedding_path
        else:
            cfg.ESM_EMB_CACHE_TRAIN = os.path.join(embeddings_dir, '420_RBPs_embeddings.npy')

        if "Probe_ID" in data.columns:
            data = data.drop("Probe_ID", axis=1)

        # Split the data into training set based on Probe_Set
        train_data = data.loc[data['Probe_Set'] == "SetA"]
        na_seq = train_data['NA_Seq'].to_list()
        train_data = train_data.drop(["NA_Seq", "Probe_Set"], axis=1).values

    elif nuc_type.upper() == "DNA":
        cfg.NA_VOCAB = 'ACGT'
        intensities_file = os.path.join(intensities_dir, "DNA_norm_data_464.csv")
        data = pd.read_csv(intensities_file)

        if 'NA_Seq' not in data.columns:
            if 'DNA_Seq' in data.columns:
                data = data.rename(columns={'DNA_Seq': 'NA_Seq'})
            else:
                print("Neither 'NA_Seq' nor 'DNA_Seq' found.")
                sys.exit(1)

        if protein_ids:
            # Filter columns based on provided protein IDs
            cols_to_keep = ['NA_Seq'] + protein_ids
            data = data[cols_to_keep]
            cfg.ESM_EMB_CACHE_TRAIN = embedding_path
        else:
            cfg.ESM_EMB_CACHE_TRAIN = os.path.join(embeddings_dir, '464_DBPs_embeddings.npy')
        
        # Split the data into training and testing sets
        train_data = data.iloc[:-10000]
        na_seq = train_data['NA_Seq'].to_list()
        train_data = train_data.drop("NA_Seq", axis=1).values


    print(train_data.shape)
    cfg.TRAIN_MATRIX = train_data
    cfg.TRAIN_NA_SEQS = na_seq

    data = TrainData(cfg=cfg)
    trainer = Trainer(data, cfg=cfg)
    model = trainer.train()

    # save model
    SAVE_MODEL_PATH = os.path.join(output_dir, f"{args.model_name}.pt")
    payload = {'model': model.state_dict(), 'cfg': cfg.__dict__,}
    torch.save(payload, SAVE_MODEL_PATH)



if __name__ == '__main__':
    main()
