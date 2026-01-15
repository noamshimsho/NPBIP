import pickle
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats


"""Multi-task CNN model for predicting nucleic acid-protein binding intensities"""

# --- Model Parameters ---
params_dict = {
    "dropout": 0.362233801349954,
    "epochs": 72,          
    "batch": 512,          
    "regu": 0.0,           
    "hidden1": 6029,
    "hidden2": 1168,
    "filters1": 2376,
    "hidden_sec": 152,
    "filters_sec": 151,
    "leaky_alpha": 0.23149394545024274,
    "filters_long_length": 24,
    "filters_long": 51
}

# Calculated dimensions for the final linear layers based on filter counts
# merge_2 represents the concatenation of the 5 main convolutional paths
params_dict["merge_2"] = params_dict['filters1'] * 4 + params_dict["filters_long"]
# output_layer is the sum of all dense hidden layers and concatenated features
params_dict["output_layer"] = (
    params_dict['hidden_sec'] +
    params_dict["hidden2"] +
    params_dict["hidden1"] +
    params_dict["merge_2"]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Sequence Encoding ---
# Standardizes the mapping for DNA (ACGT) and RNA (ACGU)
NUCLEIC_ACIDS = 'ACGT'
NN_TO_IX = {nucleotide: i for i, nucleotide in enumerate(NUCLEIC_ACIDS)}
NN_TO_IX['U'] = 3 # Map U to the same index as T for RNA/DNA compatibility

def encode_sequence(seq: str, seq_length: int = 41) -> np.ndarray:
    """
    Converts a sequence string into a one-hot encoded NumPy array (Length, 4).
    Uses 'zero' padding to match the NucProNet/MultiRBP logic.
    """
    seq = seq.upper()
    # Initialize with zeros for padding-to-zero logic
    tensor = np.zeros((seq_length, 4), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= seq_length: break
        if char in NN_TO_IX:
            tensor[i, NN_TO_IX[char]] = 1.0
    return tensor

# --- Dataset Class ---
class ArrayBasedDataset(Dataset):
    """
    Handles loading NA probes and their corresponding binding intensity targets.
    Encodes on-the-fly to minimize RAM usage during large-scale training.
    """
    def __init__(self, data_df: pd.DataFrame, seq_len: int = 41):
        self.sequences = data_df['NA_Seq'].values
        # Drop the sequence column to leave only the target binding scores for all NBPs
        self.targets = data_df.drop('NA_Seq', axis=1).values
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert sequence to one-hot array
        x_array = encode_sequence(self.sequences[idx], self.seq_len)
        # Convert targets to float32 tensor
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return torch.tensor(x_array), y

# --- Model Architecture ---
class MultiTaskModel(nn.Module):
    """
    Multi-path CNN architecture for the NewSeq task.
    Uses parallel kernels (sizes 5, 7, 9, 11, and 24) to detect motifs 
    of varying lengths simultaneously.
    """
    def __init__(self, params, input_dim, output_dim):
        super(MultiTaskModel, self).__init__()

        # Parallel Convolutional kernels for scale-invariant motif detection
        self.conv_kernel_long = nn.Conv1d(input_dim, params["filters_long"], params["filters_long_length"])
        self.conv_kernel_11 = nn.Conv1d(input_dim, params["filters1"], 11)
        self.conv_kernel_9 = nn.Conv1d(input_dim, params["filters1"], 9)
        self.conv_kernel_7 = nn.Conv1d(input_dim, params["filters1"], 7)
        self.conv_kernel_5 = nn.Conv1d(input_dim, params["filters1"], 5)
        
        # Secondary path for local 5-mer features
        self.conv_kernel_5_sec = nn.Conv1d(input_dim, params["filters_sec"], 5)

        # Fully Connected layers for feature integration
        self.hidden_dense_relu = nn.Linear(params["merge_2"], params["hidden1"])
        self.hidden_dense_relu1 = nn.Linear(params["hidden1"], params["hidden2"])
        self.hidden_dense_sec = nn.Linear(params["filters_sec"], params["hidden_sec"])
        self.output_layer = nn.Linear(params["output_layer"], output_dim)

        self.dropout = nn.Dropout(params["dropout"])
        self.leaky_relu = nn.LeakyReLU(negative_slope=params["leaky_alpha"])

    def forward(self, x):
        # Input shape: (Batch, Length, 4) -> (Batch, 4, Length) for Conv1d
        x = x.permute(0, 2, 1)

        # Multi-path Convolutions with ReLU activation
        c_long = F.relu(self.conv_kernel_long(x))
        c_11 = F.relu(self.conv_kernel_11(x))
        c_9 = F.relu(self.conv_kernel_9(x))
        c_7 = F.relu(self.conv_kernel_7(x))
        c_5 = F.relu(self.conv_kernel_5(x))
        c_5_sec = F.relu(self.conv_kernel_5_sec(x))

        # Global Max Pooling (collapses spatial dimension to find strongest motif match)
        # Using flatten(1) instead of squeeze() to ensure batch dimension is preserved
        p_long = F.max_pool1d(c_long, c_long.size(-1)).flatten(1)
        p_11 = F.max_pool1d(c_11, c_11.size(-1)).flatten(1)
        p_9 = F.max_pool1d(c_9, c_9.size(-1)).flatten(1)
        p_7 = F.max_pool1d(c_7, c_7.size(-1)).flatten(1)
        p_5 = F.max_pool1d(c_5, c_5.size(-1)).flatten(1)
        p_5_sec = F.max_pool1d(c_5_sec, c_5_sec.size(-1)).flatten(1)

        # Merge Path 1: Aggregates broad range of motif scales
        merge_a = torch.cat([p_11, p_7, p_long, p_9, p_5], dim=1)
        merge_a_drop = self.dropout(merge_a)
        dense_a = F.relu(self.hidden_dense_relu(merge_a_drop))
        dense_a_drop = self.dropout(dense_a)
        dense_b = F.relu(self.hidden_dense_relu1(dense_a_drop))

        # Merge Path 2: Processes secondary local features
        dense_sec = F.relu(self.hidden_dense_sec(self.dropout(p_5_sec)))

        # Final concatenation of all processed and raw features (Dense/Skip connection)
        final_features = torch.cat([dense_sec, dense_b, merge_a_drop, dense_a], dim=1)
        
        # Linear output layer for all tasks
        return self.leaky_relu(self.output_layer(final_features))

# --- Training Logic ---
def train(dataloader, model, loss_fn, optimizer, num_nbps):
    """
    Executes one full training pass over the dataloader.
    """
    model.train()
    total_loss = 0
    for na, y in dataloader:
        na, y = na.to(DEVICE), y.to(DEVICE).reshape(-1, num_nbps)
        
        pred = model(na)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss

# --- Evaluation Logic ---
def test(dataloader, model, num_nbps):
    """
    Evaluates the model and computes the mean Pearson correlation across all tasks.
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for na, y in dataloader:
            na, y = na.to(DEVICE), y.to(DEVICE).reshape(-1, num_nbps)
            all_preds.append(model(na).cpu())
            all_targets.append(y.cpu())
            
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # Calculate Pearson R for each individual protein task
    correlations = [stats.pearsonr(targets[:, i], preds[:, i])[0] for i in range(num_nbps)]
    mean_pearson = np.mean(correlations)
    print(f"Mean Pearson correlation: {mean_pearson:.4f}")
    return preds, targets, mean_pearson

class LogCoshLoss(nn.Module):
    """
    Implementation of the Log-Cosh loss function used in NPBIP.
    It provides a smooth regression loss that is less sensitive to outliers.
    """
    def forward(self, y_pred, y_true):
        diff = torch.clamp(y_pred - y_true, min=-80, max=80)
        return torch.mean(torch.log(torch.cosh(diff)))

def main():
    parser = argparse.ArgumentParser(description="Multi-task NewSeq Model Trainer")
    parser.add_argument("nucleic_acid_type", choices=["RNA", "DNA"], help="Specify RNA or DNA data")
    parser.add_argument("model_name", help="Output filename for saved model weights")
    parser.add_argument("--protein_ids", help="Optional file of specific protein IDs for training")
    args = parser.parse_args()

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(THIS_DIR, "../../models/SimBind")
    intensities_dir = os.path.join(THIS_DIR, "../../data/intensities")


    # Load and partition data according to experimental sets (SetA/SetB for RNA, Index for DNA)
    if args.nucleic_acid_type.upper() == "RNA":
        data = pd.read_csv(os.path.join(intensities_dir, "RNA_norm_data_420.csv"))
        # Standardize sequence column name
        if 'RNA_Seq' in data.columns:
            data = data.rename(columns={'RNA_Seq': 'NA_Seq'})
            
        train_data = data[data['Probe_Set'] == "SetA"].drop("Probe_Set", axis=1)
        test_data = data[data['Probe_Set'] == "SetB"].drop("Probe_Set", axis=1)
    else:
        data = pd.read_csv(os.path.join(intensities_dir, "DNA_norm_data_464.csv")) 
        if 'DNA_Seq' in data.columns:
            data = data.rename(columns={'DNA_Seq': 'NA_Seq'})
            
        # First 31,728 probes for training, last 10,000 for testing
        train_data = data.iloc[:-10000]
        test_data = data.iloc[-10000:]

    # Remove irrelevant metadata columns if they exist
    if "Probe_ID" in train_data.columns:
        train_data = train_data.drop("Probe_ID", axis=1)
        test_data = test_data.drop("Probe_ID", axis=1)

    # Filter by specific proteins if provided via argument
    if args.protein_ids:
        with open(args.protein_ids, "r") as f:
            ids = [line.strip() for line in f]
        train_data = train_data[['NA_Seq'] + ids]
        test_data = test_data[['NA_Seq'] + ids]

    num_nbps = train_data.shape[1] - 1
    
    # Model Initialization
    model = MultiTaskModel(params_dict, input_dim=4, output_dim=num_nbps).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = LogCoshLoss()

    # Optimized DataLoaders with parallel worker threads and pinned memory for faster GPU transfer
    train_loader = DataLoader(ArrayBasedDataset(train_data), batch_size=params_dict['batch'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ArrayBasedDataset(test_data), batch_size=4096, 
                             num_workers=4, pin_memory=True)

    print(f"Training {num_nbps} protein tasks for {params_dict['epochs']} epochs...")
    for epoch in range(params_dict['epochs']):
        epoch_loss = train(train_loader, model, loss_fn, optimizer, num_nbps)
        print(f"Epoch {epoch+1}/{params_dict['epochs']} - Total Loss: {epoch_loss:.4f}")

    # Final Evaluation & Saving
    test(test_loader, model, num_nbps)
    SAVE_MODEL_PATH = os.path.join(output_dir, f"{args.model_name}.pt")
    print(f"Saving model to {SAVE_MODEL_PATH}...")
    torch.save(model.state_dict(), SAVE_MODEL_PATH)

    # Save the order of proteins for later SimBind similarity calculations
    protein_order = list(train_data.drop("NA_Seq", axis=1).columns)
    with open(os.path.join(output_dir, f"{args.model_name}_order_protein.pkl"), "wb") as f:
        pickle.dump(protein_order, f)

if __name__ == "__main__":
    main()