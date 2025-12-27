import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import copy
import json
from sklearn.model_selection import StratifiedKFold
import argparse

# Helper Functions
def one_hot_encode(labels, num_classes=2, device='cpu'):
    """One-hot 编码"""
    return torch.eye(num_classes, device=device)[labels]

def load_data(loaded_data_path, label_path, llm_path, device):
    """Load and process data."""
    loaded_data = torch.load(loaded_data_path)
    labels_df = pd.read_csv(label_path)
    positive_genes = set(labels_df['GeneID'].dropna().apply(lambda x: int(x)))
    all_genes = loaded_data.geneid

    # GeneID and GOID nodes setup
    goid_nodes = torch.zeros(47920, dtype=torch.long).to(device)
    updated_geneid = torch.cat([torch.tensor(loaded_data.geneid, dtype=torch.long).to(device), goid_nodes], dim=0)
    loaded_data.geneid = updated_geneid

    # Labels setup
    labels = [1 if gene in positive_genes else 0 for gene in all_genes]
    label_gene = one_hot_encode(torch.tensor(labels, dtype=torch.long).to(device), device=device)

    label_goid = torch.zeros((47920, 2), dtype=torch.long).to(device)
    label = torch.cat([label_gene, label_goid], dim=0)
    loaded_data.y = label

    # Load LLM data (ISTJ predictions)
    llm = pd.read_csv(llm_path, sep=',')
    llm_dict = dict(zip(llm['GeneID'].tolist(), llm['confidence_level'].tolist()))

    # Process ISTJ predictions based on LLM data
    istj_raw = np.array([llm_dict.get(gene, np.nan) for gene in all_genes], dtype=np.float32)
    istj = np.nan_to_num(istj_raw, nan=0.0)
    istj = torch.tensor(istj, dtype=torch.float32).to(device)  # Move ISTJ predictions to device
    loaded_data.istj_predict = istj

    return loaded_data

# XunZi-M (Multi-Omics Learning) GCN Model Definition
class XunZi_M_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, goid_input_dim=4096):
        super(XunZi_M_GCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

        self.goid_dnn = nn.Sequential(
            nn.Linear(goid_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.ReLU()
        )

        self.goid_proj1 = nn.Linear(input_dim, hidden_dim1)
        self.goid_proj2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, data):
        x, edge_index, mask = data.x, data.edge_index, data.mask

        gene_x = x[mask][:, :self.input_dim]
        goid_x = x[~mask]

        with torch.no_grad():
            goid_x_transformed = self.goid_dnn(goid_x)

        x_transformed = torch.cat([gene_x, goid_x_transformed], dim=0)

        x_hidden1 = F.relu(self.conv1(x_transformed, edge_index))
        x_hidden1 = F.dropout(x_hidden1, p=0.2, training=self.training)
        x_hidden1[~mask] = self.goid_proj1(goid_x_transformed)

        x_hidden2 = F.relu(self.conv2(x_hidden1, edge_index))
        x_hidden2 = F.dropout(x_hidden2, p=0.2, training=self.training)
        x_hidden2[~mask] = self.goid_proj2(self.goid_proj1(goid_x_transformed))

        out = self.fc(x_hidden2)
        return out

# XunZi-R (Mechanistic Reasoning) Part: Integration of LLM Predictions (ISTJ)
class XunZi_R(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(XunZi_R, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, istj):
        # Adjust the predictions using ISTJ as an additional feature
        combined_input = torch.cat([x, istj.unsqueeze(1)], dim=1)
        out = self.fc(combined_input)
        return out

# Final XunZi Model: Integrates XunZi-M and XunZi-R
class XunZi(nn.Module):
    def __init__(self, xunzi_m, xunzi_r):
        super(XunZi, self).__init__()
        self.xunzi_m = xunzi_m
        self.xunzi_r = xunzi_r

    def forward(self, data):
        # Get predictions from XunZi-M (multi-omics part)
        xunzi_m_out = self.xunzi_m(data)

        # Get ISTJ predictions
        istj = data.istj_predict

        # Combine XunZi-M and XunZi-R predictions
        final_out = self.xunzi_r(xunzi_m_out, istj)
        return final_out

# Training Script for XunZi (Multi-Omics + Mechanistic Reasoning)
def train_xunzi(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    loaded_data_path = args.graph_data
    llm_path = args.llm_data
    label_path = args.label_data

    # Load data
    graph_data = load_data(loaded_data_path, label_path, llm_path, device)
    graph_data = graph_data.to(device)

    # Hyperparameters
    feature_dim = args.feature_dim
    hidden_dim1 = args.hidden_dim1
    hidden_dim2 = args.hidden_dim2
    output_dim = args.output_dim

    # Initialize XunZi-M and XunZi-R models
    xunzi_m = XunZi_M_GCN(input_dim=feature_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=output_dim,
                          goid_input_dim=4096).to(device)

    xunzi_r = XunZi_R(input_dim=hidden_dim2, output_dim=output_dim).to(device)

    # Final XunZi model
    model = XunZi(xunzi_m, xunzi_r).to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], dtype=torch.float).to(device))

    # Cross-validation setup
    kf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)

    # Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(graph_data.x.cpu().numpy(), graph_data.y.cpu().numpy())):
        print(f"\n===== Fold {fold + 1} =====")

        # Data splitting
        train_mask = torch.tensor(train_idx, dtype=torch.long).to(device)
        val_mask = torch.tensor(val_idx, dtype=torch.long).to(device)

        # Training
        model.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            out = model(graph_data)
            loss = criterion(out[train_mask], graph_data.y[train_mask])
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # Save model after training
        model_path = f'./models/model_fold_{fold + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model for fold {fold + 1} saved to {model_path}")

    print("Training complete!")

# Evaluation Script for XunZi
def evaluate_xunzi(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    loaded_data_path = args.graph_data
    llm_path = args.llm_data
    label_path = args.label_data

    # Load data
    graph_data = load_data(loaded_data_path, label_path, llm_path, device)
    graph_data = graph_data.to(device)

    # Hyperparameters
    feature_dim = args.feature_dim
    hidden_dim1 = args.hidden_dim1
    hidden_dim2 = args.hidden_dim2
    output_dim = args.output_dim

    # Initialize XunZi-M and XunZi-R models
    xunzi_m = XunZi_M_GCN(input_dim=feature_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=output_dim,
                          goid_input_dim=4096).to(device)

    xunzi_r = XunZi_R(input_dim=hidden_dim2, output_dim=output_dim).to(device)

    # Final XunZi model
    model = XunZi(xunzi_m, xunzi_r).to(device)

    # Load the saved model
    model_path = './models/model_fold_1.pth'  # Load a specific fold model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluation
    with torch.no_grad():
        out = model(graph_data)
        val_mask = graph_data.mask
        val_out = out[val_mask]
        val_labels = graph_data.y[val_mask].cpu().numpy()
        val_probs = torch.softmax(val_out, dim=1)[:, 1].cpu().numpy()

        # Calculate AUC
        auc_score = roc_auc_score(val_labels, val_probs)
        print(f"AUC for model evaluation: {auc_score:.4f}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # Save ROC curve
        output_dir = './results'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/roc_curve.png')
        print(f"ROC curve saved to {output_dir}/roc_curve.png")

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Train and Evaluate XunZi (Multi-Omics + Mechanistic Reasoning)")
    parser.add_argument('--graph_data', type=str, required=True, help="Path to graph data (e.g., .pt file)")
    parser.add_argument('--llm_data', type=str, required=True, help="Path to LLM output data")
    parser.add_argument('--label_data', type=str, required=True, help="Path to label data")
    parser.add_argument('--feature_dim', type=int, default=36, help="Feature dimension")
    parser.add_argument('--hidden_dim1', type=int, default=64, help="Hidden dimension 1")
    parser.add_argument('--hidden_dim2', type=int, default=16, help="Hidden dimension 2")
    parser.add_argument('--output_dim', type=int, default=2, help="Output dimension (number of classes)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=2e-5, help="Weight decay for optimizer")
    parser.add_argument('--k_folds', type=int, default=10, help="Number of cross-validation folds")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Train and evaluate the model
    train_xunzi(args)  # Call the training function
    evaluate_xunzi(args)  # Call the evaluation function
