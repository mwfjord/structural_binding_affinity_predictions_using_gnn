import os
import random
from statistics import stdev
import torch
import numpy as np
import wandb
import omegaconf
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch import nn
from scipy.stats import pearsonr, spearmanr

# Import your custom dataset and model modules
from dataset import PDDataset
from model import ProteinDNAGNN


class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
        self.device = device
        data_dir = config.dataset.path
        self.hyperparameters = config.hyperparameters
        
        # Initialize datasets and dataloaders
        self.train_dataset = PDDataset(data_dir, cfg=config)
        self.val_dataset = PDDataset(data_dir, validation=True, cfg=config)

        self.val_affinity_table = wandb.Table(data=[], columns=["True Binding Affinity", "Predicted Binding Affinity"])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.best_params.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.best_params.batch_size,
            shuffle=False
        )

        
        self.max_Kd = self.config.dataset.max_logKd
        self.cost_fn = self.cost
        self.criterion = nn.MSELoss()
        self.model = ProteinDNAGNN(input_dim=self.train_dataset.num_node_features, model_params=self.config_dict["best_params"], config=self.config)
        self.model = self.model.to(self.device)
        
        momentum = self.config_dict["best_params"]["sgd_momentum"]
        weight_decay = self.config_dict["best_params"]["weight_decay"]
        lr = self.config_dict["best_params"]["learning_rate"]
        _gamma = self.config_dict["best_params"]["scheduler_gamma"]

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=_gamma)

        # Print samples of dataset.
        if config.utility.verbose.data:
            for data in self.train_dataset:
                print(f"x: {data.x.shape}, edge_index: {data.edge_index.shape}", "edge_attr: ", data.edge_attr.shape, "y: ", data.y)

        # Check that edge_index and edge_attr are correct.
        for data in self.train_dataset:
            n_edges = data.edge_index.shape[1]
            if n_edges != data.edge_attr.shape[0]:
                raise ValueError(f"Number of edges in edge_index ({n_edges}) does not match number of edge features in edge_attr ({data.edge_attr.shape[0]})")


        # Early stopping parameters.
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = config.trainer.early_stopping_patience
        
        self.use_wandb = config.wandb.get("use_wandb", False)

    def cost(self, yhat, y):
        torch_eps = torch.tensor(1e-8, device=yhat.device, dtype=yhat.dtype)
        regularization_term = 1/torch.max(torch.var(yhat), torch_eps)
        lambda_reg = 0.001#* self.optimizer.param_groups[0]['lr']
        return self.criterion(yhat, y) + lambda_reg*regularization_term #sum(p.abs().sum() for p in self.model.parameters())

    
    def check_nan(self, input, output):
        if torch.isnan(output).any().item():
            print("Cause of nan", input)
            print("Output", output)
            raise ValueError("NaN detected in output.")

    def log_metrics(self, epoch, train_loss, val_loss, y_true, y_pred):

        # Ensure inputs are numpy arrays
        if hasattr(y_true, "detach"):
            y_true = y_true.detach().cpu().numpy()
        if hasattr(y_pred, "detach"):
            y_pred = y_pred.detach().cpu().numpy()

        # Compute additional metrics
        rmse = np.sqrt(val_loss)  # Assuming val_loss is MSE
        mae = np.mean(np.abs(y_true - y_pred))
        if np.std(y_pred) < 1e-6:
            print("Warning:  y_pred has very low variance. Pearson and Spearman correlation coefficients may not be reliable.")
            print("Std of y_pred:", np.std(y_pred))
            pearson_corr = 0
            spearman_corr = 0
        else:
            pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            spearman_corr, _ = spearmanr(y_true.flatten(), y_pred.flatten())

        data = [[true, pred] for true, pred in zip(y_true.flatten(), y_pred.flatten())]
        self.val_affinity_table = wandb.Table(data=data, columns=["True Binding Affinity", "Predicted Binding Affinity"])
        lr = self.optimizer.param_groups[0]['lr']
        if self.use_wandb:
            wandb.log({
                "epoch": epoch,
                "lr": lr,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/rmse": rmse,
                "val/mae": mae,
                "val/pearson_corr": pearson_corr,
                "val/spearman_corr": spearman_corr,
            })

        

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, data in enumerate(tqdm(self.train_loader, desc="Training", leave=False, unit="it")):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                out = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
                self.check_nan(data, out)
                out = out.squeeze()  
                data.y = data.y.squeeze()
                loss = self.cost_fn(out, data.y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
        self.scheduler.step()

        return total_loss / num_batches

    def validate_epoch(self, epoch, train_loss):
        self.model.eval()
        total_loss = 0.0
        all_true = torch.tensor([], dtype=torch.float32)
        all_pred = torch.tensor([], dtype=torch.float32)
        all_pred = all_pred.to(self.device)
        all_true = all_true.to(self.device)
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="Validation", leave=False):
                data = data.to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                    out = self.model(data.x, data.edge_attr , data.edge_index, data.batch)
                    out = out.squeeze()
                    data.y = data.y.squeeze()
                    loss = self.cost_fn(out, data.y)
                    all_true = torch.cat([all_true, data.y])
                    all_pred = torch.cat([all_pred, out])
                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_loader)

        self.log_metrics(epoch=epoch, 
            train_loss=train_loss,
            val_loss=avg_val_loss,
            y_true=all_true,  
            y_pred=all_pred)  
        
        return avg_val_loss
    

    def save_checkpoint(self, epoch, path="./trained_models"):
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def create_model_artifact(self, checkpoint_path):
        artifact = wandb.Artifact(name="graph_classification_model", type="model")
        artifact.add_file(checkpoint_path)
        return artifact


    def run(self):
        epochs = self.config.trainer.epochs
        log_interval = self.config.logging.log_interval
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)

            if epoch % log_interval == 0:
              val_loss = self.validate_epoch(epoch, train_loss)

              print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

              # Checkpointing and early stopping.
              if val_loss < self.best_val_loss:
                  self.best_val_loss = val_loss
                  checkpoint_path = self.save_checkpoint(epoch)
                  if self.use_wandb:
                    wandb.log_artifact(self.create_model_artifact(checkpoint_path))
                  self.early_stop_counter = 0 
              else:
                  self.early_stop_counter += 1
                  if self.early_stop_counter >= self.early_stop_patience:
                      print("Early stopping triggered.")
                      break   
                  
        if self.use_wandb:
            scatter = wandb.plot.scatter(self.val_affinity_table, x="True Binding Affinity", y="Predicted Binding Affinity", title="True vs Predicted Binding Affinity")
            wandb.log({
                    "val/true_vs_pred": scatter
                    })

 

    







