from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.pairwise2 import format_alignment
blosum62 = substitution_matrices.load("BLOSUM62")
import re
from Bio import SeqIO, pairwise2
import argparse
import sys
import csv
import os
import multiprocessing as mp
from tqdm import tqdm

def compute_AA_SID(seq1, seq2):
    """
    Aligns two domain sequences using Needleman–Wunsch global alignment
    with BLOSUM62 (gap open: -11, gap extension: -1) and calculates
    the AA SID (number of exact identical positions divided by alignment length).
    
    Parameters:
        seq1, seq2: domain sequences (strings)
    
    Returns:
        sid: A float representing the AA SID.
    """
    # Perform global alignment with BLOSUM62, gap open penalty -11, gap extension penalty -1.
    # pairwise2.align.globalds returns a list of alignments; we take the first (best) alignment.
    alignments = pairwise2.align.globalds(seq1, seq2, blosum62, -11, -1)
    
    aln1, aln2, score, start, end = alignments[0]
    
    # Calculate the number of identical positions.
    identical = sum(1 for a, b in zip(aln1, aln2) if a == b)
    sid = identical / len(aln1)
    return sid

def calculate_pairwise_AA_SID(rbr1, rbr2):
    """
    Calculates the pairwise AA SID for two RNA binding regions (RBRs).
    
    Each RBR is represented as a list of domain sequences (with flanks).
    
    For RBRs with the same number of domains:
        The pairwise AA SID is the mean AA SID computed by aligning
        each corresponding domain (e.g., RRM1 vs. RRM1, RRM2 vs. RRM2).
    
    For RBRs with different numbers of domains:
        The shorter RBR is slid over the longer RBR in all possible contiguous ways,
        but only allowing alignments where adjacent domains are aligned.
        For each possible alignment, the mean AA SID is computed.
        The maximum mean AA SID across these alignments is returned.
    
    Parameters:
        rbr1, rbr2: Lists of domain sequences (strings) representing each RBR.
    
    Returns:
        pairwise_sid: A float representing the pairwise AA SID.
    """
    n1 = len(rbr1)
    n2 = len(rbr2)
    if n1 == 0 or n2 == 0:
        raise ValueError("One of the RBRs is empty!")
    
    # If both RBRs have the same number of domains, align them one-to-one.
    if n1 == n2:
        sids = [compute_AA_SID(rbr1[i], rbr2[i]) for i in range(n1)]
        return sum(sids) / len(sids)
    else:
        # Identify the shorter and longer RBR.
        if n1 < n2:
            shorter, longer = rbr1, rbr2
        else:
            shorter, longer = rbr2, rbr1
        
        max_mean_sid = 0.0
        # The number of possible contiguous alignments is:
        # len(longer) - len(shorter) + 1.
        for offset in range(len(longer) - len(shorter) + 1):
            #print(offset)
            sids = []
            # For each domain in the shorter RBR, align with the corresponding domain
            # in the longer RBR (starting at position 'offset').
            for i in range(len(shorter)):
                sid = compute_AA_SID(shorter[i], longer[i + offset])
                sids.append(sid)
            mean_sid = sum(sids) / len(sids)
            if mean_sid > max_mean_sid:
                max_mean_sid = mean_sid
        return max_mean_sid
    
def parse_domain_fasta_file(fasta_file):
    protein_domains = {}

    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.id  # header without the '>' character
        protein_id = header.split('_')[0]

        # Extract the "from" coordinate from the header using a regex: e.g., _from(\d+)_to
        m_from = re.search(r"_from(\d+)_to", header)
        from_coord = int(m_from.group(1)) if m_from else 0

        # Create a tuple (from_coord, sequence) to help sort later.
        domain_seq = str(record.seq)
        if protein_id in protein_domains:
            protein_domains[protein_id].append((from_coord, domain_seq))
        else:
            protein_domains[protein_id] = [(from_coord, domain_seq)]

    # Sort the domain sequences for each protein by the "from" coordinate.
    for pid in protein_domains:
        protein_domains[pid].sort(key=lambda x: x[0])
        # Replace list of tuples with list of sequences (ordered by position)
        protein_domains[pid] = [seq for _, seq in protein_domains[pid]]

    return protein_domains

# Function to run in each process
def compute_sid(args):
    query_id, query_seq, train_id, train_seq = args
    sid = calculate_pairwise_AA_SID(query_seq, train_seq)
    return {
        'query_protein': query_id,
        'train_protein': train_id,
        'similarity_score': sid
    }

    
def main():
    parser = argparse.ArgumentParser(description="calculate similarity scores between proteins")
    parser.add_argument("query_fasta_domain", help="Path to fasta file with domain sequences for query proteins")
    parser.add_argument("nucleic_acid_type", help="RNA or DNA")
    parser.add_argument("run_id", type=str, help="run ID for saving/loading files")

    args = parser.parse_args()

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(THIS_DIR, "../../output") 
    train_protein_dir = os.path.join(THIS_DIR, "../../data/sequences") 

    run_id = args.run_id
    os.makedirs(os.path.join(output_dir, run_id), exist_ok=True)
    nuc_type = args.nucleic_acid_type.upper()
    query_fasta_domain = args.query_fasta_domain


    if nuc_type == "RNA":
        train_fasta_domain = os.path.join(train_protein_dir, "Train_RBPs_domains_plus_15.fasta")
    elif nuc_type == "DNA":
        train_fasta_domain = os.path.join(train_protein_dir, "Train_DBPs_domains_plus_15.fasta")
    else:
        print("error must specified nucleic_acid_type DNA or RNA")
        sys.exit(1) 

    query_binding_region = parse_domain_fasta_file(query_fasta_domain) # dictionary{}


    if len(query_binding_region) == 0:
        print("There are no proteins with binding domains...")
        sys.exit(1)
    else:
        print(f"for {len(query_binding_region)} there are binding domains!")

    train_binding_region = parse_domain_fasta_file(train_fasta_domain)


    # Step 3: Compute pairwise AA SID for all protein pairs.
    # Prepare all combinations of query and train
    arg_list = [
    (query_id, query_seq, train_id, train_seq)
    for query_id, query_seq in query_binding_region.items()
    for train_id, train_seq in train_binding_region.items()]

    with mp.Pool(mp.cpu_count()) as pool:
        # Use imap (not map) so tqdm can show progress
        results = list(tqdm(pool.imap(compute_sid, arg_list), total=len(arg_list)))


    # Step 4: Save the results to a CSV file.
    pairwise_output_path = os.path.join(output_dir, run_id ,f"{nuc_type}_pairwise_AA_SID_results.csv")
    with open(pairwise_output_path, 'w', newline='') as csvfile:
        fieldnames = ['query_protein', 'train_protein', 'similarity_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Saved pairwise AA SID results pairs to {pairwise_output_path}.")


if __name__ == "__main__":
    main()
