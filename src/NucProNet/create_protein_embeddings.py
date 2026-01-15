import os
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
from Bio import SeqIO
import sys


def read_fasta_full(fasta_file):
    sequences = []
    protein_ids = []
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        # Each seq_record object contains id, description, and sequence
        sequences.append(str(seq_record.seq)) # Extract the sequence as a string
        protein_ids.append(str(seq_record.id))
    print(f"Extracted {len(sequences)} sequences.")

    return sequences, protein_ids


def compute_esm2_embeddings(seqs: List[str], batch_size: int, device: torch.device) -> np.ndarray:
    hf_model_name = "facebook/esm2_t48_15B_UR50D"
    print(f"[ESM2] Computing embeddings with {hf_model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModel.from_pretrained(
        hf_model_name,
        torch_dtype=(torch.float16 if device.type=='cuda' else torch.float32)
    ).to(device).eval()
    print(f"[ESM2] Model embedding dimension: {model.config.hidden_size}")

    outs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
            hs = model(**inputs).last_hidden_state            # [B, L, D]
            am = inputs["attention_mask"]                     # [B, L]

            pooled = []
            for j in range(hs.size(0)):
                seq_len = int(am[j].sum().item())
                if seq_len > 2:
                    aa = hs[j, 1:seq_len-1, :]                # exclude <cls> and <eos>
                else:
                    aa = hs[j, 1:seq_len, :]
                pooled.append(aa.mean(0).float().cpu().numpy())
            outs.append(np.stack(pooled, axis=0))

            print(f"  ESM2 {min(i+batch_size, len(seqs))}/{len(seqs)}")
            if device.type == 'cuda' and (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()
    E = np.concatenate(outs, axis=0).astype(np.float32)
    print(E.shape)
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    return E

ESM_EMB_BATCH: int = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main():
    parser = argparse.ArgumentParser(description="Create protein embeddings")
    parser.add_argument("fasta", help="Path to input protein FASTA file")
    parser.add_argument("run_id", type=str, help="run ID for saving/loading files")
    args = parser.parse_args()

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(THIS_DIR, "../../output") 

    run_id = args.run_id
    os.makedirs(os.path.join(output_dir, run_id), exist_ok=True)

    fasta_file = args.fasta
    protein_sequences, proteins_ids = read_fasta_full(fasta_file)

    if not protein_sequences:
        print("No protein sequennces found...")
        sys.exit(1)

    # Compute ESM2 embeddings
    E = compute_esm2_embeddings(protein_sequences, ESM_EMB_BATCH,device)
    # Save embeddings
    emb_file = os.path.join(output_dir, run_id, "esm2_emb.npy")
    np.save(emb_file, E)
    print(f"Saved ESM2 embeddings to {emb_file} with shape {E.shape}")

if __name__ == "__main__":
    main()