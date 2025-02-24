import omegaconf
import torch
import wandb
import random
from trainer import Trainer
from tuner import BaysianTuner
import numpy as np

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    # Load configuration from a YAML file using OmegaConf.
    config = omegaconf.OmegaConf.load("../utils/config.yaml")
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    use_wandb = config.wandb.use_wandb
    
  
    if use_wandb:
      config.wandb.name += f"_{random.randint(0, 1000)}"
      wandb.login()
      wandb.init(
          project=config.wandb.project,
          config=config_dict,
        #   name=config.wandb.name,
          reinit=True
      )


    seed = config.utility.get("seed", 42)
    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:3")
        print("Using GPU 3")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create and run the trainer.
    if config.utility.train:
        trainer = Trainer(config, device)
        trainer.run()   
    else:
        tuner = BaysianTuner(config)
        tuner.tune_hyperparameters()


    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
