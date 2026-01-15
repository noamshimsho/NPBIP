import argparse
import subprocess
import sys # Import sys for printing to stderr and exiting
import os

def main():

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="SimBInd full pipeline")
    parser.add_argument("protein_query_fasta", help="Path to query protein FASTA file")
    parser.add_argument("nuc_query_fasta", help="Path to query nucleic-acid FASTA file")
    parser.add_argument("nucleic_acid_type", help="RNA or DNA")
    parser.add_argument("run_id", type=str, default=None, help="Optional run ID for saving output files")
    
    args = parser.parse_args()

    run_id = args.run_id

    output_dir = os.path.join(SCRIPT_DIR, "../../output", run_id)
    os.makedirs(output_dir, exist_ok=True)


    protein_query_fasta = args.protein_query_fasta
    nuc_query_fasta = args.nuc_query_fasta

    if not os.path.isfile(protein_query_fasta):
        print(f"Error: File '{protein_query_fasta}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(nuc_query_fasta):
        print(f"Error: File '{nuc_query_fasta}' does not exist.", file=sys.stderr)
        sys.exit(1)

    nuc_type = args.nucleic_acid_type.upper()
    if nuc_type not in ["RNA", "DNA"]:
        print("error must specified nucleic_acid_type DNA or RNA")
        sys.exit(1)

    
    protein_domain_script = os.path.join(SCRIPT_DIR, "protein_domain.py")
    pair_wise_script = os.path.join(SCRIPT_DIR, "pair_wise.py")
    NewSeq_prediction_script = os.path.join(SCRIPT_DIR, "NewSeq_prediction.py")
    NewProNewSeq_agg_script = os.path.join(SCRIPT_DIR, "NewProNewSeq_agg.py")

    #protein domain command
    command_domain = ["python", protein_domain_script, protein_query_fasta, nuc_type, run_id]
    #pairwise command
    query_fasta_domain = os.path.join(output_dir, f"{nuc_type}_domains_plus_15.fasta")
    command_pairwise = ["python", pair_wise_script, query_fasta_domain, nuc_type, run_id]
    #NewSeq prediction command
    command_new_seq_prediction = ["python", NewSeq_prediction_script, nuc_query_fasta, nuc_type, run_id]
    #NewProNewSeq_agg command
    new_seq_prediction = os.path.join(output_dir, f"{nuc_type}_final_new_seq_predictions.npy")
    similarity_path = os.path.join(output_dir, f"{nuc_type}_pairwise_AA_SID_results.csv")
    command_new_pro_new_seq_prediction = ["python", NewProNewSeq_agg_script, 
                                          new_seq_prediction, similarity_path, nuc_query_fasta, nuc_type, run_id]
    # no_identical_similarity = args.no_identical_similarity
    # if no_identical_similarity:  # some Python boolean
    #     command_new_pro_new_seq_prediction.append("--no_identical_similarity")


    print(f"Run command: {' '.join(command_domain)}", file=sys.stderr) # Good for debugging
    try:
        
        result_domain = subprocess.run(command_domain,capture_output=True,text=True,check=True)
        print("--- protein_domain.py STDOUT ---")
        print(result_domain.stdout)
        print("\n\n")

        print(f"Run command: {' '.join(command_pairwise)}", file=sys.stderr) # Good for debugging
        result_pairwise = subprocess.run(command_pairwise, capture_output=True,text=True,check=True)
        print("--- pair_wise.py STDOUT ---")
        print(result_pairwise.stdout)
        print("\n\n")

        print(f"Run command: {' '.join(command_new_seq_prediction)}", file=sys.stderr) # Good for debugging
        result_new_seq_prediction = subprocess.run(command_new_seq_prediction, capture_output=True,text=True,check=True)
        print("--- NewSeq_prediction.py STDOUT ---")
        print(result_new_seq_prediction.stdout)
        print("\n\n")


        print(f"Run command: {' '.join(command_new_pro_new_seq_prediction)}", file=sys.stderr) # Good for debugging
        result_new_pro_new_seq_final_prediction = subprocess.run(command_new_pro_new_seq_prediction, capture_output=True,text=True,check=True)
        print("--- NewProNewSeq_final_prediction.py STDOUT ---")
        print(result_new_pro_new_seq_final_prediction.stdout)
        print("\n\n")


    except subprocess.CalledProcessError as e:
        print(f"Error: A script failed with exit code {e.returncode}.", file=sys.stderr)

        print("\n--- STDOUT ---", file=sys.stderr)
        print(e.stdout if e.stdout else "No standard output captured.", file=sys.stderr)

        print("\n--- STDERR ---", file=sys.stderr)
        print(e.stderr if e.stderr else "No standard error captured.", file=sys.stderr)
        # The actual error message from protein_domain.py will likely be in e.stderr

        sys.exit(1) # Exit main_NPBIP.py with an error status
    except FileNotFoundError:
        # This error occurs if "python" or "protein_domain.py" cannot be found
        print(f"Error: Could not find 'python' or the script 'protein_domain.py'.", file=sys.stderr)
        print(f"Make sure 'protein_domain.py' is in the same directory or in your PATH.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
