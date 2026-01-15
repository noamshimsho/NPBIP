import argparse
import subprocess
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr

def get_unique_run_id(base_dir, run_id):

    full_path = os.path.join(base_dir, run_id)
    if not os.path.exists(full_path):
        return run_id

    i = 1
    while True:
        new_run_id = f"{run_id}_{i}"
        new_path = os.path.join(base_dir, new_run_id)
        if not os.path.exists(new_path):
            return new_run_id
        i += 1


def main():

    parser = argparse.ArgumentParser(description="NPBIP full pipeline")
    parser.add_argument("protein_query_fasta", help="Path to query protein FASTA file")
    parser.add_argument("nuc_query_fasta", help="Path to query nucleic-acid FASTA file")
    parser.add_argument("nucleic_acid_type", help="RNA or DNA")
    parser.add_argument("run_id", type=str, default=None, help="run ID for saving output files")
    parser.add_argument("--intensities_csv", help = "The true intensities of query proteins and nucleic acids")
    args = parser.parse_args()

    intensities = args.intensities_csv
    if args.run_id is None:
        run_id = "output"
    else:
        run_id = args.run_id

    run_id = get_unique_run_id("output", run_id)
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(THIS_DIR, "output", run_id)
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


    # --- Setup Commands ---
    command_NucProNet = ["python", "src/NucProNet/NucProNet.py", protein_query_fasta, 
                         nuc_query_fasta, nuc_type, run_id]
    
    command_SimBind = ["python", "src/SimBind/SimBind.py", protein_query_fasta, 
                       nuc_query_fasta, nuc_type, run_id]

    processes = []
    
    try:
        # 1. Start NucProNet (Process 1)
        print(f"Starting NucProNet command: {' '.join(command_NucProNet)}", file=sys.stderr)
        process_NucProNet = subprocess.Popen(
            command_NucProNet, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True # Decode output as text
        )
        processes.append(process_NucProNet)
        
        # 2. Start SimBind (Process 2)
        print(f"Starting SimBind command: {' '.join(command_SimBind)}", file=sys.stderr)
        process_SimBind = subprocess.Popen(
            command_SimBind, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        processes.append(process_SimBind)
        
        # --- WAIT FOR BOTH PROCESSES TO FINISH ---
        
        # NucProNet Results
        stdout_NPN, stderr_NPN = process_NucProNet.communicate()
        if process_NucProNet.returncode != 0:
            raise subprocess.CalledProcessError(
                process_NucProNet.returncode, 
                command_NucProNet, 
                stdout=stdout_NPN, 
                stderr=stderr_NPN
            )
        print("--- NucProNet.py STDOUT ---")
        print(stdout_NPN)
        print("\n\n")

        # SimBind Results
        stdout_SB, stderr_SB = process_SimBind.communicate()
        if process_SimBind.returncode != 0:
            raise subprocess.CalledProcessError(
                process_SimBind.returncode, 
                command_SimBind, 
                stdout=stdout_SB, 
                stderr=stderr_SB
            )
        print("--- SimBind.py STDOUT ---")
        print(stdout_SB)
        print("\n\n")

        print("✅ Both NucProNet and SimBind finished successfully. Starting next step...")
        
        df_simbind = pd.read_csv(f"output/{run_id}/SimBind_predictions.csv", index_col=0)
        df_nucpronet = pd.read_csv(f"output/{run_id}/NucProNet_predictions.csv", index_col=0)

        df_simbind_zscore = df_simbind.apply(zscore, axis=1, ddof=1)
        df_nucpronet_zscore = df_nucpronet.apply(zscore, axis=1, ddof=1)

        NPBIP_res = (df_simbind_zscore + df_nucpronet_zscore) / 2.0
        NPBIP_res.to_csv(f"output/{run_id}/NPBIP_predictions.csv", index=True)

        print(f"✅ NPBIP predictions saved to 'output/{run_id}/NPBIP_predictions.csv'.")

        if intensities:
            # Compute Pearson correlation between true intensities and NPBIP/NucProNet/SimBind predictions
            correlation_results = {}
            p_names = NPBIP_res.index.tolist()
            true_intensities = pd.read_csv(intensities)
            for p_name in p_names:
                if p_name in true_intensities:
                    corr_npbip, _ = pearsonr(true_intensities[p_name], NPBIP_res.loc[p_name])
                    corr_nucpronet, _ = pearsonr(true_intensities[p_name], df_nucpronet.loc[p_name])
                    corr_simbind, _ = pearsonr(true_intensities[p_name], df_simbind.loc[p_name])
                    correlation_results[p_name] = {
                        "NPBIP": corr_npbip,
                        "NucProNet": corr_nucpronet,
                        "SimBind": corr_simbind
                    }
                else:
                    correlation_results[p_name] = None

            # Save correlation results
            corr_df = pd.DataFrame.from_dict(correlation_results, orient='index')
            print(f"✅ Pearson correlation results:\n{corr_df}")

            corr_df.to_csv(f"output/{run_id}/correlation_results.csv", index=False)
            print(f"✅ Pearson correlation results saved to 'output/{run_id}/correlation_results.csv'.")


    except subprocess.CalledProcessError as e:
        print(f"Error: A script failed with exit code {e.returncode}.", file=sys.stderr)
        print(f"Command: {' '.join(e.cmd)}", file=sys.stderr) # Print the failing command

        print("\n--- STDOUT ---", file=sys.stderr)
        print(e.stdout if e.stdout else "No standard output captured.", file=sys.stderr)

        print("\n--- STDERR ---", file=sys.stderr)
        print(e.stderr if e.stderr else "No standard error captured.", file=sys.stderr)
        
        sys.exit(1)
    
    except FileNotFoundError:
        # Check if the error was due to 'python' or the script path
        print(f"Error: Could not find 'python' or one of the scripts (NucProNet.py or SimBind.py).", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()