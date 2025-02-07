import os
import random
import torch
import numpy as np
import wandb
import omegaconf
from tqdm import tqdm
from torch import nn
from torch_geometric.loader import DataLoader

# Import your custom dataset and model modules
from proteinDNADataset import ProteinDNADataset
from model import ProteinDNAGNN


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    """
    Exponential Moving Average (EMA) for model weights.
    Maintains a smoothed version of the model parameters.
    """
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach() 
            for name, param in model.named_parameters() if param.requires_grad
        }

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Initialize datasets and dataloaders
        self.train_dataset = ProteinDNADataset("../data")
        self.val_dataset = ProteinDNADataset("../data", validation=True)
        self.test_dataset = ProteinDNADataset("../data", test=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False
        )

        # Initialize the model.
        # Note: Adjust `input_dim` if necessary; here we use the dataset's num_node_features.
        self.model = ProteinDNAGNN(
            input_dim=self.train_dataset.num_node_features,
            hidden_dim=config.model.hidden_units
        ).to(device)

        # Set up optimizer and loss function.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate)
        self.criterion = nn.MSELoss()

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
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize EMA for model weights.
        self.ema = EMA(self.model, decay=config.training.get("ema_decay", 0.99))

        # Early stopping parameters.
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = config.training.get("early_stop_patience", 10)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, data in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = self.model(data.x, data.edge_index, data.batch)
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
                with torch.cuda.amp.autocast():
                    out = self.model(data.x, data.edge_index, data.batch)
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

    def run(self):
        epochs = self.config.training.epochs
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch()

            # Log metrics to Weights & Biases.
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
                wandb.log_artifact(self.create_model_artifact(checkpoint_path))
                self.early_stop_counter = 0  # Reset counter when improvement is seen.
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    print("Early stopping triggered.")
                    break

        # Optionally apply EMA weights before final evaluation or saving.
        self.ema.apply_shadow(self.model)
        wandb.finish()


def main():
    # Login to Weights & Biases.
    wandb.login()

    # Load configuration from a YAML file using OmegaConf.
    config = omegaconf.OmegaConf.load("config.yaml")
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)

    # Initialize a new W&B run.
    wandb.init(
        project=config.wandb.project,
        config=config_dict,
        name=config.wandb.name,
        reinit=True
    )

    # Set random seed for reproducibility.
    seed = config.training.get("seed", 42)
    set_seed(seed)

    # Choose device. Adjust this if you want to use a specific GPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Log dataset details as a W&B artifact.
    train_dataset = ProteinDNADataset("../data")
    val_dataset = ProteinDNADataset("../data", validation=True)
    test_dataset = ProteinDNADataset("../data", test=True)
    data_details = {
        "num_node_features": train_dataset.num_node_features,
        "num_edge_features": train_dataset.num_edge_features,
        "num_classes": train_dataset.num_classes,
        "num_training_samples": len(train_dataset),
        "num_validation_samples": len(val_dataset),
        "num_test_samples": len(test_dataset)
    }
    dataset_artifact = wandb.Artifact(name="KLF1_K_d", type="dataset", metadata=data_details)
    dataset_artifact.add_dir("../data")
    wandb.log_artifact(dataset_artifact)

    # Create and run the trainer.
    trainer = Trainer(config, device)
    trainer.run()


if __name__ == "__main__":
    main()
