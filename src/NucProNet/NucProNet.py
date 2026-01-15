import argparse
import subprocess
import sys
import os


def main():

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="NucProNet full pipeline")
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

    
    embedding_script = os.path.join(SCRIPT_DIR, "create_protein_embeddings.py")
    predict_script = os.path.join(SCRIPT_DIR, "predict.py")

    #create embedding command
    command_embedding = ["python", embedding_script, protein_query_fasta, run_id]
    #predict command
    protein_embedding = os.path.join(output_dir, "esm2_emb.npy")
    command_predict = ["python", predict_script, protein_query_fasta, protein_embedding, 
                       nuc_query_fasta, nuc_type, run_id]
    

    print(f"Run command: {' '.join(command_embedding)}", file=sys.stderr) # Good for debugging
    try:
        
        result = subprocess.run(command_embedding, capture_output=True, text=True, check=True)
        print("--- create_protein_embeddings.py STDOUT ---")
        print(result.stdout)
        print("\n\n")

        print(f"Run command: {' '.join(command_predict)}", file=sys.stderr) # Good for debugging
        result = subprocess.run(command_predict, capture_output=True,text=True,check=True)
        print("--- NucProNet_predict.py STDOUT ---")
        print(result.stdout)
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
