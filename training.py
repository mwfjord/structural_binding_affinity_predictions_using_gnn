import torch
import torch_geometric
import gemmi
import Bio
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader



print("PyTorch:", torch.__version__)
print("PyTorch Geometric:", torch_geometric.__version__)
print("GEMMI:", gemmi.__version__)
print("Biopython:", Bio.__version__)


class ProteinDNA_GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, output_dim=1):
        super(ProteinDNA_GNN, self).__init__()
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Fully Connected (MLP) for regression
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Aggregate node embeddings into a graph-level embedding
        x = global_mean_pool(x, batch)
        
        # Final regression output (binding affinity)
        x = self.fc(x)
        return x
      
      

# Load pre-saved datasets
train_dataset = torch.load("./data/train_dataset.pt")
val_dataset = torch.load("./data/val_dataset.pt")
test_dataset = torch.load("/data/test_dataset.pt")

# Create PyTorch Geometric DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

print("Dataloaders ready!")
