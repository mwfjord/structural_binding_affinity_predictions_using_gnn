import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp, TopKPooling, Linear
import torch.nn as nn

class ProteinDNAGNN_mini(torch.nn.Module):
    def __init__(self, input_dim, model_params, config):
        """
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of the output (default is 1 for regression).
            cfg (optional): Configuration object with model parameters.
        """
        super(ProteinDNAGNN, self).__init__()
        hidden_channels = int(model_params["model_embedding_size"])
        self.conv1 = GCNConv(input_dim, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm3 = nn.BatchNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels, int(hidden_channels/2))
        self.lin2 = Linear(int(hidden_channels/2), 1)

    def forward(self,x, edge_attr, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm3(x)

        

        # 2. Readout layer
        x = gmp(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        
        return x
    


class ProteinDNAGNN(torch.nn.Module):
    def __init__(self, input_dim, model_params, config):
        """
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of the output (default is 1 for regression).
            cfg (optional): Configuration object with model parameters.
        """
        super(ProteinDNAGNN, self).__init__()

        # Retrieve model configuration or set defaults.
        self.num_layers = model_params["model_layers"]
        embedding_size = model_params["model_embedding_size"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        # dropout_rate = model_params["model_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]
        self.prediction_threshold = config["dataset"]["max_logKd"]

        self.conv1 = GCNConv(input_dim, embedding_size)
        self.transf1 = nn.Linear(embedding_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)

        self.conv_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()


        for i in range(self.num_layers):
            self.conv_layers.append(GCNConv(embedding_size, embedding_size))
            self.dense_layers.append(Linear(embedding_size, embedding_size))
            self.bn_layers.append(nn.BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        self.linear1 = Linear(embedding_size*2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))  
        self.linear3 = Linear(int(dense_neurons/2), 1)  

    def forward(self,x, edge_attr, edge_index, batch_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        global_representation = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = torch.relu(self.dense_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x , edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                    )
                # Add current representation
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
        
        x = sum(global_representation)
        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)
        # x = torch.relu(x)
        # tensor_threshold = torch.tensor([self.prediction_threshold], dtype=torch.float32)
        # tensor_threshold = tensor_threshold.to(x.device)
        # x = torch.min(tensor_threshold, x)
        return x