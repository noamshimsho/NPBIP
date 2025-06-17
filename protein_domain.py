from Bio import SeqIO
import subprocess
import argparse
import os
import shutil
import sys

def read_fasta(file_path):
    """
    Reads a FASTA file and returns a list of sequences.

    :param file_path: Path to the FASTA file.
    :return: List of SeqRecord objects.
    """
    
    records = list(SeqIO.parse(file_path, "fasta"))
    print(f"Successfully read {len(records)} sequences from {file_path}.")
    return records

    
def run_hmmscan(fasta_file, hmm_db, output):
    """
    Runs hmmscan against a protein FASTA file using the specified HMM database.
    
    :param fasta_file: Path to the protein FASTA file.
    :param hmm_db: Path to the HMM database file (e.g., rbd_models.hmm).
    :param output_tbl: Path for the hmmscan tabular output.
    """
    command = [
        "hmmscan",
        "-E", "0.01",         # Full sequence E-value cutoff
        "--domE", "0.01",      # Domain conditional E-value cutoff
        "--domtblout", output, # Output file in tabular format
        hmm_db,
        fasta_file
    ]

    try:
        print("Running hmmscan...")
        print(command)
        subprocess.run(command, check=True)
        print(f"hmmscan finished; results written to {output}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running hmmscan: {e}")

def load_protein_sequences(fasta_file):
    """
    Load protein sequences from a FASTA file into a dictionary.
    
    :param fasta_file: Path to the FASTA file.
    :return: Dictionary mapping sequence ID to sequence string.
    """
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Use record.id as key (if your FASTA headers contain spaces or extra info,
        # you may need to adjust this to extract the proper identifier).
        sequences[record.id] = str(record.seq)
    return sequences

def parse_domtblout(domtblout_file, fasta_file, add_15 = True):
    """
    Parse a hmmscan domtblout file to extract, for each protein,
    all domain occurrences (with coordinates and optionally the domain sequence).
    
    :param domtblout_file: Path to the domtblout file.
    :param fasta_file: (Optional) Path to the FASTA file with the protein sequences.
                       If provided, the function will extract the domain sequence using the alignment coordinates.
    :return: Dictionary mapping query protein ID to a list of domain hit dictionaries.
             Each hit dictionary contains keys such as:
             'domain_name', 'ali_from', 'ali_to', 'cEvalue', 'iEvalue', 'score', 'bias', 'domain_seq'
             (if fasta_file is provided, otherwise 'domain_seq' is an empty string).
    """
    # If a FASTA file is provided, load the sequences.
    protein_seqs = {}
    if fasta_file:
        protein_seqs = load_protein_sequences(fasta_file)
    
    results = {}
    
    with open(domtblout_file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue  # skip comment/header lines
            # Split the line by whitespace.
            parts = line.strip().split()

            #print(parts)
            
            domain_name = parts[0]
            query_name = parts[3]
            try:
                cEvalue = float(parts[11])
                # Use the alignment coordinates (columns 17 and 18).
                # Note: domtblout coordinates are 1-indexed.
                ali_from = int(parts[19])
                ali_to   = int(parts[20])
            except ValueError:
                # Skip line if any conversion fails.
                continue

            
            # Extract the domain sequence from the query protein sequence if available,
            # extending 15 amino acids to each side (if possible).
            domain_seq = ""
            if protein_seqs and query_name in protein_seqs:
                full_seq = protein_seqs[query_name]
                # Convert domtblout positions (1-indexed) to Python's 0-indexed.
                # Extend 15 amino acids upstream and downstream.
                if add_15:
                    start = max(0, ali_from - 1 - 15)
                    end = min(len(full_seq), ali_to + 15)
                else:
                    start = max(0, ali_from - 1)
                    end = min(len(full_seq), ali_to)
                domain_seq = full_seq[start:end]
            
            hit = {
                'domain_name': domain_name,
                'ali_from': ali_from,
                'ali_to': ali_to,
                'cEvalue': cEvalue,
                'domain_seq': domain_seq
            }
            
            # Append this hit to the results for the query protein.
            if query_name not in results:
                results[query_name] = []
            results[query_name].append(hit)

    
    return results


def write_fasta(domains_dict, output_file):
    """
    Write domain sequences to a FASTA file.
    """
    with open(output_file, "w") as fout:
        for protein_id, domain_list in domains_dict.items():
            for idx, domain in enumerate(domain_list, start=1):
                header = (f">{protein_id}_{domain['domain_name']}_"
                          f"domain{idx}_from{domain['ali_from']}_to{domain['ali_to']}_"
                          f"cEvalue:{domain['cEvalue']:.2e}")
                fout.write(header + "\n")
                seq = domain['domain_seq']
                for i in range(0, len(seq), 80):
                    fout.write(seq[i:i+80] + "\n")
    print(f"[INFO] Saved {output_file}")





def main():
    parser = argparse.ArgumentParser(description="Find RNA/DNA-binding domains using HMMER.")
    parser.add_argument("fasta", help="Path to input protein FASTA file")
    parser.add_argument("nucleic_acid_type", help="RNA/DNA")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    fasta_file = args.fasta
    nuc_type = args.nucleic_acid_type

    domtblout_path = os.path.join("output", F"hmmscan_results_{nuc_type}.tbl")
    fasta_out_path = os.path.join("output", F"{nuc_type}_domains_plus_15.fasta") 

    protein_sequences = read_fasta(fasta_file)
    # Print out the first sequence as an example
    if protein_sequences:
        print("First sequence record:")
        print(f"ID: {protein_sequences[0].id}")
        print(f"Description: {protein_sequences[0].description}")
        print(f"Sequence: {protein_sequences[0].seq}")
    else:
        print("No protein sequennces found...")
        sys.exit(1)

    if nuc_type == "RNA":
        hmm_db = "/home/noam/programs/combined_rna_binding.hmm"
    elif nuc_type == "DNA":
        hmm_db = "/home/noam/programs/combined_dna_binding.hmm"
    else:
        print("error must specified nucleic_acid_type DNA or RNA")
        sys.exit(1)        
    run_hmmscan(fasta_file, hmm_db, domtblout_path)
    domain_hits = parse_domtblout(domtblout_path, fasta_file, add_15=True)
    write_fasta(domain_hits, fasta_out_path)

if __name__ == "__main__":
    # Check if hmmscan is installed
    if shutil.which("hmmscan") is None:
        print("Error: 'hmmscan' is not installed or not found in your system PATH.")
        print("Please install HMMER and ensure 'hmmscan' is accessible from the command line.")
        sys.exit(1)
    main()