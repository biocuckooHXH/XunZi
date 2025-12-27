import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pathlib
from torchinfo import summary

# General settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define feature aggregation module
class FeatureAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features):
        if features.size(0) == 1:
            return features[0]
        weights = F.softmax(self.mlp(features), dim=0)
        aggregated_features = torch.sum(weights * features, dim=0)
        return aggregated_features


# Define the GCN model with aggregator
class GCNWithAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, aggregator, goid_input_dim=4096, goid_output_dim=198):
        super(GCNWithAggregator, self).__init__()
        self.aggregator = aggregator
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.goid_dnn = nn.Sequential(
            nn.Linear(goid_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, goid_output_dim),
            nn.ReLU()
        )
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, data):
        x, edge_index, mask = data.x, data.edge_index, data.mask
        gene_x = x[mask][:, :198]  # Only gene features
        goid_x = x[~mask]  # GOID node features
        goid_x_transformed = self.goid_dnn(goid_x)
        x_transformed = torch.cat([gene_x, goid_x_transformed], dim=0)

        x = F.relu(self.conv1(x_transformed, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x


# Helper functions
def one_hot_encode(labels, num_classes=2):
    return torch.eye(num_classes)[labels]


def load_data(loaded_data_path, label_path, allgene_path, columns_to_keep, llm_path):
    loaded_data = torch.load(loaded_data_path)

    subset = loaded_data.x[:12816, :]
    mask1 = torch.ones(subset.shape[1], dtype=torch.bool)
    mask1[columns_to_keep] = False
    subset[:, mask1] = 0
    loaded_data.x[:12816, :] = subset

    labels1 = pd.read_csv(label_path)
    positive_genes = set(labels1.iloc[:, 0].dropna())
    df = pd.read_csv(allgene_path)
    all_genes = df['GeneID'].tolist()
    labels = [1 if geneid in positive_genes else 0 for geneid in all_genes]

    label_gene = one_hot_encode(torch.tensor(labels, dtype=torch.long))
    label_goid = torch.zeros((47920, 2), dtype=torch.long)
    label = torch.cat([label_gene, label_goid], dim=0)
    loaded_data.y = label

    llm = pd.read_csv(llm_path, sep='\t')
    llm_dict = dict(zip(llm['GeneID'].tolist(), llm['confidence_level'].tolist()))

    istj_raw = np.array([llm_dict.get(gene, np.nan) for gene in all_genes], dtype=np.float32)
    istj = np.nan_to_num(istj_raw, nan=0.0)
    istj = torch.tensor(istj, dtype=torch.long)
    loaded_data.istj_predict = istj

    return loaded_data


# Train the model for XunZi
def train_xunzi(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    loaded_data_path = args.graph_data
    llm_path = args.llm_data
    label_path = args.label_data
    allgene_path = args.allgene_data

    # Load data
    graph_data = load_data(loaded_data_path, label_path, allgene_path, args.columns_to_keep, llm_path)
    graph_data = graph_data.to(device)

    # Hyperparameters
    feature_dim = args.feature_dim
    hidden_dim1 = args.hidden_dim1
    hidden_dim2 = args.hidden_dim2
    output_dim = args.output_dim

    # Initialize model
    aggregator = FeatureAggregator(input_dim=feature_dim, hidden_dim=128, output_dim=feature_dim)
    model = GCNWithAggregator(input_dim=feature_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=output_dim,
                              aggregator=aggregator).to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], dtype=torch.float).to(device))

    # Cross-validation setup
    kf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)

    # Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(graph_data.x[graph_data.mask].cpu().numpy(), graph_data.y.cpu().numpy())):
        print(f"Fold {fold + 1}")

        # Data splitting
        train_mask = torch.tensor(train_idx, dtype=torch.long).to(device)
        test_mask = torch.tensor(val_idx, dtype=torch.long).to(device)

        # Training
        model.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            out = model(graph_data)
            loss = criterion(out[train_mask], graph_data.y[train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    out_val = out[test_mask]
                    auc_val = roc_auc_score(graph_data.y[test_mask].cpu().numpy(), out_val.cpu().numpy())
                    print(f"Epoch {epoch}, Validation AUC: {auc_val:.4f}")

        # Save model after each fold
        model_path = f"./models/fold_{fold + 1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model for fold {fold + 1} saved to {model_path}")


# Evaluation of the model
def evaluate_xunzi(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    loaded_data_path = args.graph_data
    llm_path = args.llm_data
    label_path = args.label_data
    allgene_path = args.allgene_data

    # Load data
    graph_data = load_data(loaded_data_path, label_path, allgene_path, args.columns_to_keep, llm_path)
    graph_data = graph_data.to(device)

    # Hyperparameters
    feature_dim = args.feature_dim
    hidden_dim1 = args.hidden_dim1
    hidden_dim2 = args.hidden_dim2
    output_dim = args.output_dim

    # Initialize model
    aggregator = FeatureAggregator(input_dim=feature_dim, hidden_dim=128, output_dim=feature_dim)
    model = GCNWithAggregator(input_dim=feature_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=output_dim,
                              aggregator=aggregator).to(device)

    # Load the best model
    model_path = './models/fold_1.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluation
    with torch.no_grad():
        out = model(graph_data)
        val_mask = graph_data.mask
        out_val = out[val_mask]
        auc_val = roc_auc_score(graph_data.y[val_mask].cpu().numpy(), out_val.cpu().numpy())
        print(f"Validation AUC: {auc_val:.4f}")


if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Train and Evaluate XunZi (Finetuning for Disease Prediction)")
    parser.add_argument('--graph_data', type=str, required=True, help="Path to graph data (e.g., .pth file)")
    parser.add_argument('--llm_data', type=str, required=True, help="Path to LLM output data")
    parser.add_argument('--label_data', type=str, required=True, help="Path to label data")
    parser.add_argument('--allgene_data', type=str, required=True, help="Path to all genes data")
    parser.add_argument('--columns_to_keep', type=list, required=True, help="Columns to keep in data")
    parser.add_argument('--feature_dim', type=int, default=36, help="Feature dimension")
    parser.add_argument('--hidden_dim1', type=int, default=128, help="Hidden dimension 1")
    parser.add_argument('--hidden_dim2', type=int, default=64, help="Hidden dimension 2")
    parser.add_argument('--output_dim', type=int, default=2, help="Output dimension (number of classes)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=2e-5, help="Weight decay for optimizer")
    parser.add_argument('--k_folds', type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Train and evaluate the model
    train_xunzi(args)  # Call the training function
    evaluate_xunzi(args)  # Call the evaluation function
