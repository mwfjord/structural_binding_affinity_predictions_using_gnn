import os
import random
from matplotlib.pylab import f
import torch
import numpy as np
import wandb
import omegaconf
from tqdm import tqdm
from torch import nn
from torch_geometric.loader import DataLoader
from mango import scheduler, Tuner
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from zmq import device

# Import your custom dataset and model modules
from dataset import PDDataset
from model import ProteinDNAGNN


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
        self.device = device
        data_dir = config.dataset.path
        self.hyperparameters = config.parameters
        
        # Initialize datasets and dataloaders
        self.train_dataset = PDDataset(data_dir, cfg=config)
        self.val_dataset = PDDataset(data_dir, validation=True, cfg=config)
        self.test_dataset = PDDataset(data_dir, test=True, cfg=config)
        
        self.max_Kd = self.config.dataset.max_Kd

        # Print samples of dataset.
        if config.utility.verbose:
            for data in self.train_dataset:
                print(f"x: {data.x.shape}, edge_index: {data.edge_index.shape}")


        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hyperparameters.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.hyperparameters.batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.hyperparameters.batch_size,
            shuffle=False
        )

        # Initialize the model.
        # Note: Adjust `input_dim` if necessary; here we use the dataset's num_node_features.

        self.model = ProteinDNAGNN(
            input_dim=self.train_dataset.num_node_features,
            hidden_dim=config.model.hidden_units,
            cfg=config
        ).to(device)

        # Set up optimizer and loss function.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.training.learning_rate)
        self.criterion = self.RMSELoss
        # self.criterion = self.PearsonCorrelation

        # Main learning rate scheduler: CosineAnnealingWarmRestarts.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.training.get("T_0", 10),
            T_mult=config.training.get("T_mult", 2),
            eta_min=config.training.get("eta_min", 1e-6)
        )

        # Optional warmup scheduler.
        self.warmup_epochs = config.training.get("warmup_epochs", 0)
        if self.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=self.warmup_epochs
            )
        else:
            self.warmup_scheduler = None

        # Initialize AMP scaler for mixed precision training.
        self.scaler = torch.amp.GradScaler()

        # Initialize EMA for model weights.
        self.ema = EMA(self.model, decay=config.training.get("ema_decay", 0.99))

        # Early stopping parameters.
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = config.training.get("early_stop_patience", 10)
        
        self.use_wandb = config.wandb.get("use_wandb", False)

    def RMSELoss(self, yhat,y):
        return torch.sqrt(torch.mean((yhat-y)**2))
    
    def PearsonCorrelation(self, yhat, y):
        mean_yhat = torch.mean(yhat)
        mean_y = torch.mean(y)
        std_yhat = torch.std(yhat)
        std_y = torch.std(y)
        return torch.mean((yhat - mean_yhat) * (y - mean_y) / (std_yhat * std_y))

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, data in enumerate(tqdm(self.train_loader, desc="Training", leave=False, unit="it")):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            # print("This is the data: ", data)

            with torch.amp.autocast(device_type='cuda'):
                out = self.model(data.x, data.edge_index, data.batch)
                out = out.squeeze()  # Remove unnecessary dimension for MSE loss.
                data.y = data.y.squeeze()
                loss = self.criterion(out, data.y)

            self.scaler.scale(loss).backward()

            # Gradient clipping to stabilize training.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update EMA weights after each optimizer step.
            self.ema.update(self.model)

            total_loss += loss.item()

            # Scheduler update:
            if self.warmup_scheduler is not None and epoch <= self.warmup_epochs:
                # During warmup, update the warmup scheduler per batch.
                self.warmup_scheduler.step()
            else:
                # Update the cosine annealing scheduler using fractional epoch progress.
                current_iter = epoch - 1 + (batch_idx + 1) / num_batches
                self.scheduler.step(current_iter)

        return total_loss / num_batches

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="Validation", leave=False):
                data = data.to(self.device)
                with torch.amp.autocast(device_type='cuda'):
                    out = self.model(data.x, data.edge_index, data.batch)
                    out = out.squeeze()  # Remove unnecessary dimension for MSE loss.
                    data.y = data.y.squeeze()
                    loss = self.criterion(out, data.y)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, path="./trained_models"):
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def create_model_artifact(self, checkpoint_path):
        artifact = wandb.Artifact(name="graph_classification_model", type="model")
        artifact.add_file(checkpoint_path)
        return artifact
    
    def update_params_to_best(self):
        best_params = self.config_dict["best_params"]
        self.model = ProteinDNAGNN(input_dim=self.train_dataset.num_node_features, model_params=best_params)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=best_params["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=best_params["scheduler_gamma"])

    def run_tuning(self, params):

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=params["batch_size"],
            shuffle=True
        )
        self.val_loader = DataLoader(
                self.validation_dataset,
                batch_size=params["batch_size"],
                shuffle=False
            )
    
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = ProteinDNAGNN(input_dim=self.train_dataset.num_node_features, model_params=model_params) 
        model = model.to(device)

        # < 1 increases precision, > 1 recall
        self.optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=params["learning_rate"],
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=params["scheduler_gamma"])
        
        # Start training
        self.run()
        return self.best_val_loss

    def tune_hyperparameters(self):

        HYPERPARAMETERS = self.config_dict["hyperparameters"]
        tuner_config = self.config_dict["tuner"]

        tuner = Tuner(HYPERPARAMETERS, 
            objective=self.run_tuning,
            conf_dict=tuner_config) 
        results = tuner.minimize()

        best_params = results["best_params"]
        #Write the results to a file
        with open("config.yaml", "a") as f:
            omegaconf.OmegaConf.save(best_params, f)



    def run(self):
        epochs = self.config.training.epochs
        log_interval = self.config.logging.log_interval
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)

            if epoch % log_interval == 0:
              val_loss = self.validate_epoch()
              # Log metrics to Weights & Biases.
              if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": self.optimizer.param_groups[0]["lr"]
                })

              print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

              # Checkpointing and early stopping.
              if val_loss < self.best_val_loss:
                  self.best_val_loss = val_loss
                  checkpoint_path = self.save_checkpoint(epoch)
                  if self.use_wandb:
                    wandb.log_artifact(self.create_model_artifact(checkpoint_path))
                  self.early_stop_counter = 0  # Reset counter when improvement is seen.
              else:
                  self.early_stop_counter += 1
                  if self.early_stop_counter >= self.early_stop_patience:
                      print("Early stopping triggered.")
                      break

        if self.use_wandb:
            wandb.finish()  
        


# def RMSELoss(yhat,y):
#     loss = torch.sqrt(torch.mean((yhat-y)**2))
#     return loss

# def calculate_metrics(preds, labels, epoch, mode):
#     pearson_corr = np.corrcoef(preds, labels)[0, 1]
#     rmse = np.sqrt(np.mean((preds - labels) ** 2))
#     mae = np.mean(np.abs(preds - labels))
#     # print(f"Epoch {epoch} | {mode} RMSE: {rmse:.4f}, MAE: {mae:.4f}, Pearson Correlation: {pearson_corr:.4f}")

# device = torch.device("cuda:3")
# def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
#     # Enumerate over the data
#     all_preds = np.array([])
#     all_labels = np.array([])
#     running_loss = 0.0
#     step = 0
#     for _, batch in enumerate(tqdm(train_loader)):
#         # Use GPU
#         batch.to(device)  
#         # Reset gradients
#         optimizer.zero_grad() 
#         # Passing the node features and the connection info
#         pred = model(batch.x.float(), 
#                                 batch.edge_attr.float(),
#                                 batch.edge_index, 
#                                 batch.batch) 
#         if torch.isnan(pred).any():
#             # print("Prediction is nan")
#             # print(pred)
#             # Set to zero
#             for i in range(len(pred)):
#                 if torch.isnan(pred[i]):
#                     pred[i] = 0

#         # Calculating the loss and gradients
#         loss = loss_fn(torch.squeeze(pred), batch.y.float())
#         loss.backward()  
#         optimizer.step()  
#         # Update tracking
#         running_loss += loss.item()
#         step += 1
#         pred = torch.squeeze(pred)
#         # batch.y = torch.unsqueeze(batch.y, 1)
#         all_preds = np.append(all_preds, np.rint((pred).cpu().detach().numpy()))
#         all_labels = np.append(all_labels, batch.y.cpu().detach().numpy())
#         calculate_metrics(all_preds, all_labels, epoch, "train")
#     return running_loss/step

# def test(epoch, model, test_loader, loss_fn):
#     all_preds = []
#     all_labels = []
#     running_loss = 0.0
#     step = 0
#     for batch in test_loader:
#         batch.to(device)  
#         pred = model(batch.x.float(), 
#                         batch.edge_attr.float(),
#                         batch.edge_index, 
#                         batch.batch) 
#         if torch.isnan(pred).any():
#             # print("Prediction is nan")
#             # print(pred)
#             # Set to zero
#             for i in range(len(pred)):
#                 if torch.isnan(pred[i]):
#                     pred[i] = 0

#         loss = loss_fn(torch.squeeze(pred), batch.y.float())

#          # Update tracking
#         running_loss += loss.item()
#         step += 1
#         pred = torch.squeeze(pred)
#         # batch.y = torch.unsqueeze(batch.y, 1)
#         all_preds = np.append(all_preds, np.rint((pred).cpu().detach().numpy()))
#         all_labels = np.append(all_labels, batch.y.cpu().detach().numpy())
    
#     calculate_metrics(all_preds, all_labels, epoch, "test")
#     return running_loss/step

# def run_for_parameter_tuning(params):
#     params = params[0]
#     # print(params)
#     # Load configuration from a YAML file using OmegaConf.
#     device = torch.device("cuda:3")
#     config = omegaconf.OmegaConf.load("config.yaml")
#     config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
#     data_dir = config.dataset.path

#     train_dataset = PDDataset(data_dir, cfg=config)
#     validation_dataset = PDDataset(data_dir,cfg=config)
#     train_loader = DataLoader(
#             train_dataset,
#             batch_size=params["batch_size"],
#             shuffle=True
#         )
#     val_loader = DataLoader(
#             validation_dataset,
#             batch_size=params["batch_size"],
#             shuffle=False
#         )
    

#     model_params = {k: v for k, v in params.items() if k.startswith("model_")}
#     model = ProteinDNAGNN(input_dim=train_dataset.num_node_features, model_params=model_params) 
#     model = model.to(device)
#     # < 1 increases precision, > 1 recall
#     loss_fn = RMSELoss
#     optimizer = torch.optim.SGD(model.parameters(), 
#                                 lr=params["learning_rate"],
#                                 momentum=params["sgd_momentum"],
#                                 weight_decay=params["weight_decay"])
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])
    
#     # Start training
#     best_loss = 1000
#     early_stopping_counter = 0
#     for epoch in range(300): 
#         if early_stopping_counter <= 10: # = x * 5 
#             # Training
#             model.train()
#             loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
#             # print(f"Epoch {epoch} | Train Loss {loss}")

#             # Testing
#             model.eval()
#             if epoch % 5 == 0:
#                 loss = test(epoch, model, val_loader, loss_fn)
#                 # print(f"Epoch {epoch} | Test Loss {loss}")
                
#                 # Update best loss
#                 if float(loss) < best_loss:
#                     best_loss = loss
#                     # Save the currently best model 
#                     early_stopping_counter = 0
#                 else:
#                     early_stopping_counter += 1

#             scheduler.step()
#         else:
#             print("Early stopping due to no improvement.")
#             return [best_loss]
        
#     # print(f"Finishing training with best test loss: {best_loss}")
#     # check for nan
#     if torch.isnan(torch.tensor(best_loss)):
#         print("Best loss is nan")
#         return [1000.0]
#     return [best_loss]



def main():

    # Load configuration from a YAML file using OmegaConf.
    config = omegaconf.OmegaConf.load("config.yaml")
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    use_wandb = config.wandb.use_wandb
    
    if use_wandb:
    # Login to Weights & Biases.
      wandb.login()
      # Initialize a new W&B run.
      wandb.init(
          project=config.wandb.project,
          config=config_dict,
          name=config.wandb.name,
          reinit=True
      )


    # Set random seed for reproducibility.
    seed = config.utility.get("seed", 42)
    set_seed(seed)

    # Choose device. Adjust this if you want to use a specific GPU.
    if torch.cuda.is_available():
        device = torch.device("cuda:3")
        print("Using GPU 3")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create and run the trainer.
    trainer = Trainer(config, device)
    trainer.tune_hyperparameters()
    trainer.update_params_to_best()
    

    # trainer.run()


if __name__ == "__main__":
    main()





