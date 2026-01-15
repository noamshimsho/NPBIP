import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
import csv
from scipy.special import softmax
import pandas as pd
import pickle

def get_fasta_headers(fasta_filepath: str):
    """
    Reads a FASTA file and returns a list of all sequence headers (IDs).

    The header is the string immediately following the '>' character on a
    definition line, up to the first whitespace or the end of the line.
    """
    headers = []
    try:
        with open(fasta_filepath, 'r') as f_in:
            for line in f_in:
                line = line.strip()  # Remove leading/trailing whitespace
                if line.startswith(">"):
                    # The header is the part after '>' up to the first space,
                    # or the whole line if no space.
                    header_part = line[1:] # Remove the '>'
                    # Split by the first whitespace to get only the ID part
                    # Some FASTA headers have descriptions after the first space.
                    # If you want the entire line after '>', use: headers.append(header_part)
                    header_id = header_part.split(None, 1)[0] # None splits on any whitespace
                    headers.append(header_id)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{fasta_filepath}' was not found.")
    except IOError as e:
        raise IOError(f"Error reading file '{fasta_filepath}': {e}")
    return headers


def similarity_score_sort(csv_file):

    pairwise_dict = {}
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_protein = row["query_protein"]
            train_protein = row["train_protein"]
            # Convert Pairwise_AA_SID to float.
            try:
                sid = float(row["similarity_score"])
            except ValueError:
                sid = None  # or you can choose to skip rows that don't have a valid number
            
            # Initialize list for Protein1 if not already present.
            if query_protein not in pairwise_dict:
                pairwise_dict[query_protein] = []

            # Append tuple (Protein2, Pairwise_AA_SID) to the list.
            pairwise_dict[query_protein].append((train_protein, sid))

    # Sort the list of tuples for each protein by Pairwise_AA_SID.
    for protein1 in pairwise_dict:
        # Sort in ascending order. If you want descending, add reverse=True.
        pairwise_dict[protein1].sort(key=lambda tup: tup[1], reverse=True)

    return pairwise_dict

def final_prediction(aa_sid_dict, new_seq_prediction, train_protein_order, k=20, scale=10, no_identical_similarity=False):

    res = {}
    for i, (protein_test, similarity_scores) in enumerate(aa_sid_dict.items()):
        sid_scores = []
        preds = []
        for similar_name, similar_score in similarity_scores:
            if no_identical_similarity and similar_score >= 0.99:
                #print(f"no_identical_similarity is {no_identical_similarity} {similar_name} with {similar_score} is out!")
                continue
            if similar_name not in train_protein_order:
                continue
            sid_scores.append(similar_score)
            proxy_index = train_protein_order.index(similar_name)
            pred = new_seq_prediction[:,proxy_index]
            preds.append(pred)
            if len(sid_scores) >= k:
                break

        preds = np.stack(preds)
        weights = softmax((np.array(sid_scores) * scale))
        weighted_predictions = weights[:, np.newaxis] * preds
        final_predictions_for_probes = np.sum(weighted_predictions, axis=0)

        res[protein_test] = final_predictions_for_probes

    return res
    

def main():
    parser = argparse.ArgumentParser(description="Final step predict score for query protein and query nucleic acid seq, based on weighted average of the predicted binding intensities by the calculated similarities")
    
    parser.add_argument("new_seq_prediction", help="path to the new seq prediction npy file")
    parser.add_argument("similarity_path", help="path to the similarity csv file")
    parser.add_argument("query_nuc_fasta", help="fasta file of the nucleic acids sequences")
    parser.add_argument("nucleic_acid_type", help="RNA or DNA")
    parser.add_argument("run_id", type=str, help="run ID for saving/loading files")
    parser.add_argument("--order_model_output", type=str, help="Path to the order of the protein the model output. one protein per line")
    args = parser.parse_args()

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(THIS_DIR, "../../output") 
    new_seq_models_dir = os.path.join(THIS_DIR, "../../models/SimBind") 
    run_id = args.run_id
    os.makedirs(os.path.join(output_dir, run_id), exist_ok=True)
    
    #no_identical_similarity = args.no_identical_similarity
    nuc_type = args.nucleic_acid_type.upper()

    if args.order_model_output:
        with open(args.order_model_output, "r") as f:
            model_protein_output_order = [line.strip() for line in f.readlines()]

    elif nuc_type == "RNA":
        order_protein_path = os.path.join(new_seq_models_dir,"MultiRBP_order_protein.pkl")
        with open(order_protein_path, "rb") as f:
            model_protein_output_order = pickle.load(f)
        
    elif nuc_type == "DNA":
        order_protein_path = os.path.join(new_seq_models_dir,"MultiDBP_order_protein.pkl")
        with open(order_protein_path, "rb") as f:
            model_protein_output_order = pickle.load(f)
    else:
        print("error must specified nucleic_acid_type DNA or RNA")
        sys.exit(1)

    query_nuc_fasta = args.query_nuc_fasta
    query_nuc_ids = get_fasta_headers(query_nuc_fasta)

    new_seq_prediction = np.load(args.new_seq_prediction)
    print(new_seq_prediction.shape)

    similarity_path = args.similarity_path
    pairwise_dict = similarity_score_sort(similarity_path)

    res = final_prediction(pairwise_dict, new_seq_prediction, model_protein_output_order, k=20, scale=10)


    df_final = pd.DataFrame(res).T
    df_final.columns = query_nuc_ids
    csv_file_path = os.path.join(output_dir, run_id ,f"SimBind_predictions.csv")
    df_final.to_csv(csv_file_path, index=True) 

    print(f"Successfully saved data to '{csv_file_path}'")


if __name__ == "__main__":
    main()
