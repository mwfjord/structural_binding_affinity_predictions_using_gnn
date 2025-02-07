import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

class ProteinDNAGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, cfg=None):
        """
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of the output (default is 1 for regression).
            cfg (optional): Configuration object with model parameters.
        """
        super(ProteinDNAGNN, self).__init__()
        self.cfg = cfg

        # Retrieve model configuration or set defaults.
        num_layers = cfg.model.layers if (cfg is not None and hasattr(cfg, "model") and "layers" in cfg.model) else 3
        activation_str = cfg.model.activation if (cfg is not None and hasattr(cfg, "model") and "activation" in cfg.model) else "relu"

        # Choose activation function based on configuration.
        if activation_str.lower() == "relu":
            self.activation = F.relu
        elif activation_str.lower() == "tanh":
            self.activation = torch.tanh
        elif activation_str.lower() == "leakyrelu":
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

        # Build the GCN layers.
        self.convs = nn.ModuleList()
        # The first layer transforms the input features to hidden_dim.
        self.convs.append(GCNConv(input_dim, hidden_dim))
        # Additional layers: hidden_dim -> hidden_dim.
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Fully connected layer for regression output.
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # Pass through the sequence of GCN layers.
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        
        # Global pooling to obtain a graph-level representation.
        x = global_mean_pool(x, batch)
        
        # Return the regression output (e.g., binding affinity prediction).
        return self.fc(x)
