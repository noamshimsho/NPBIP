import argparse
import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict
from Bio import SeqIO

params_dict = {
    "dropout": 0.362233801349954,
    "epochs": 78,#78
    "batch" : 512, #4096, 1024
    "regu": 0.0, # 5.7215002041656515e-06
    "hidden1" : 6029,
    "hidden2" : 1168,
    "filters1" : 2376,
    "hidden_sec" : 152,
    "filters_sec" : 151,
    "leaky_alpha" : 0.23149394545024274,
    "filters_long_length" : 24,
    "filters_long" : 51
}
params_dict["merge_2"] = params_dict['filters1'] * 4 + params_dict["filters_long"]
params_dict["output_layer"] = params_dict['hidden_sec'] + params_dict["hidden2"] + params_dict["hidden1"] + params_dict["merge_2"]


class MultiModel(nn.Module):
    def __init__(self, params_dict, input_dim, output_dim):
        super(MultiModel, self).__init__()

        # Convolutional layers
        self.conv_kernel_long = nn.Conv1d(
            in_channels=input_dim, out_channels=params_dict["filters_long"],
            kernel_size=params_dict["filters_long_length"], 
            bias=True, padding=0
        )

        self.conv_kernel_11 = nn.Conv1d(
            in_channels=input_dim, out_channels=params_dict["filters1"],
            kernel_size=11, bias=True, padding=0
        )

        self.conv_kernel_9 = nn.Conv1d(
            in_channels=input_dim, out_channels=params_dict["filters1"],
            kernel_size=9, bias=True, padding=0
        )

        self.conv_kernel_7 = nn.Conv1d(
            in_channels=input_dim, out_channels=params_dict["filters1"],
            kernel_size=7, bias=True, padding=0
        )

        self.conv_kernel_5 = nn.Conv1d(
            in_channels=input_dim, out_channels=params_dict["filters1"],
            kernel_size=5, bias=True, padding=0
        )

        self.conv_kernel_5_sec = nn.Conv1d(
            in_channels=input_dim, out_channels=params_dict["filters_sec"],
            kernel_size=5, bias=True, padding=0
        )

        # Fully connected layers
        self.hidden_dense_relu = nn.Linear(params_dict["merge_2"], params_dict["hidden1"])
        self.hidden_dense_relu1 = nn.Linear(params_dict["hidden1"], params_dict["hidden2"])
        self.hidden_dense_sec = nn.Linear(params_dict["filters_sec"], params_dict["hidden_sec"])
        
        self.output_layer = nn.Linear(params_dict["output_layer"], output_dim)
        
        self.dropout_merge_2 = nn.Dropout(params_dict["dropout"])
        self.dropout_hidden_dense_relu = nn.Dropout(params_dict["dropout"])
        self.dropout_down = nn.Dropout(params_dict["dropout"])
        self.leaky_relu = nn.LeakyReLU(negative_slope=params_dict["leaky_alpha"])

    def forward(self, x):
        # x shape: (batch_size, 41, 9)
        x = x.permute(0, 2, 1)  # PyTorch Conv1D expects (batch_size, channels, length)

        # Convolutional layers
        conv_long = F.relu(self.conv_kernel_long(x))
        conv_11 = F.relu(self.conv_kernel_11(x))  # (batch_size,filters1, 31)
        conv_9 = F.relu(self.conv_kernel_9(x))
        conv_7 = F.relu(self.conv_kernel_7(x))
        conv_5 = F.relu(self.conv_kernel_5(x))
        conv_5_sec = F.relu(self.conv_kernel_5_sec(x))
  
        pool_long = F.max_pool1d(conv_long, kernel_size=conv_long.size(-1))
        pool_11 = F.max_pool1d(conv_11, kernel_size=conv_11.size(-1))
        pool_9 = F.max_pool1d(conv_9, kernel_size=conv_9.size(-1))
        pool_7 = F.max_pool1d(conv_7, kernel_size=conv_7.size(-1))
        pool_5 = F.max_pool1d(conv_5, kernel_size=conv_5.size(-1))
        pool_5_sec = F.max_pool1d(conv_5_sec, kernel_size=conv_5_sec.size(-1))


        # first_path
        merge_first_path = torch.cat([pool_11, pool_7, pool_long, pool_9, pool_5], dim=1)
        merge_first_path_flatten = merge_first_path.squeeze()
        merge_first_path_flatten_dropout = self.dropout_merge_2(merge_first_path_flatten)
        merge_first_path_flatten_dropout_dense_a = F.relu(self.hidden_dense_relu(merge_first_path_flatten_dropout))
        merge_first_path_flatten_dropout_dense_a_dropout = self.dropout_hidden_dense_relu(merge_first_path_flatten_dropout_dense_a)
        merge_first_path_flatten_dropout_dense_a_dropout_dense_b = F.relu(self.hidden_dense_relu1(merge_first_path_flatten_dropout_dense_a_dropout))

        # second_path
        second_path_flatten = pool_5_sec.squeeze()
        second_path_flatten_dropout = self.dropout_down(second_path_flatten)
        second_path_flatten_dropout_dense_a = F.relu(self.hidden_dense_sec(second_path_flatten_dropout))


        merge_final = torch.cat([second_path_flatten_dropout_dense_a, 
                                 merge_first_path_flatten_dropout_dense_a_dropout_dense_b,
                                 merge_first_path_flatten_dropout,
                                 merge_first_path_flatten_dropout_dense_a], dim=1)

        # Output layer
        y = self.output_layer(merge_final)
        y = self.leaky_relu(y)

        return y
    
def read_fasta(file_path):
    """
    Reads a FASTA file and returns a list of sequences.

    :param file_path: Path to the FASTA file.
    :return: List of SeqRecord objects.
    """
    try:
        records = list(SeqIO.parse(file_path, "fasta"))
        print(f"Successfully read {len(records)} sequences from {file_path}.")
        return records
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def seq_to_indices(sequence: str, NN_TO_IX, NUCLEIC_ACIDS) -> List[int]:
    try:
        return [NN_TO_IX[nucleotide.upper()] for nucleotide in sequence]
    except KeyError as e:
        raise ValueError(f"Unknown nucleotide {e} in sequence '{sequence}'. Allowed: {NUCLEIC_ACIDS}")

def encode_batch(nuc_type: str, rna_sequences: List[str], target_seq_length: int = 41, padding_type: str = 'zero'):

    """
    Encodes a batch of sequences into a single one-hot encoded NumPy array.
    Returns:A 3D NumPy     
    """
    if nuc_type == "RNA":
        NUCLEIC_ACIDS: str = 'ACGU'
    else:
        NUCLEIC_ACIDS: str = 'ACGT'
    NUM_NUCLEIC_ACIDS: int = len(NUCLEIC_ACIDS)
    NN_TO_IX: Dict[str, int] = {nucleotide: i for i, nucleotide in enumerate(NUCLEIC_ACIDS)}

    batch_indices: List[List[int]] = [seq_to_indices(rna, NN_TO_IX, NUCLEIC_ACIDS) for rna in rna_sequences]
    batch_size = len(batch_indices)
    target_seq_length = max(target_seq_length, max([len(idx_list) for idx_list in batch_indices]))


    # Initialize the batch tensor based on padding_type
    if padding_type == 'zero':
        batch_tensor = np.zeros(
            (batch_size, target_seq_length, NUM_NUCLEIC_ACIDS), dtype=np.float32)
    else:
        raise ValueError(f"Unknown padding_type: '{padding_type}'. Choose 'zero' or 'equal'.")

    # Populate the batch tensor
    for i, single_seq_indices in enumerate(batch_indices):
        for j, idx in enumerate(single_seq_indices):
            batch_tensor[i,j,idx] = 1

    return batch_tensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Predicts binding intensities of nucleic acids seq to training proteins")
    parser.add_argument("nucleic_acid_type", help="RNA or DNA")
    parser.add_argument("fasta", help="fasta file of the nucleic acids sequences")
    args = parser.parse_args()

    # Create the model
    nuc_type = args.nucleic_acid_type.upper()
    if nuc_type == "RNA":
        weights_PATH = "DATA/NewSeqModels/MultiRBP.pt"
        model = MultiModel(params_dict, input_dim=4, output_dim=244).to(device)
        model.load_state_dict(torch.load(weights_PATH, map_location=device))

    elif nuc_type == "DNA":
        weights_PATH = "DATA/NewSeqModels/MultiDBP.pt"
        model = MultiModel(params_dict, input_dim=4, output_dim=562).to(device)
        model.load_state_dict(torch.load(weights_PATH, map_location=device))
    else:
        print("error must specified nucleic_acid_type DNA or RNA")  
        sys.exit(1)
    model.eval()

    # query sequences
    fasta_file = args.fasta
    query_sequences = read_fasta(fasta_file)
    batch_size = 2048
    batch_indices = range(0, len(query_sequences), batch_size)


    print("Starting prediction...")
    all_outputs = []
    for i in tqdm(batch_indices, desc="Processing Batches", total=len(batch_indices)):
    #for i in tqdm(range(0, len(query_sequences), batch_size), desc="Processing Batches"):
        batch_data_chunk = query_sequences[i : i + batch_size]
        batch_data_chunk_one_hot = encode_batch(nuc_type, batch_data_chunk, target_seq_length=params_dict['filters_long_length'])
        input_tensor = torch.tensor(np.stack(batch_data_chunk_one_hot))
        #print(input_tensor.shape, "input")

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
        all_outputs.append(output.cpu())
        
    final_predictions_tensor = torch.cat(all_outputs, dim=0)
    print(f"\nFinished. Final predictions tensor shape: {final_predictions_tensor.shape}")

    # output path
    output_npy_file = f"output/{nuc_type}_final_new_seq_predictions.npy"
    final_predictions_numpy = final_predictions_tensor.cpu().numpy()
    np.save(output_npy_file, final_predictions_numpy)
    print(f"Predictions saved as NumPy binary to: {output_npy_file}")


if __name__ == "__main__":
    main()