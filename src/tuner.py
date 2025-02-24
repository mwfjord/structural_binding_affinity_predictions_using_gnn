from trainer import Trainer
import omegaconf
import torch
from torch_geometric.data import DataLoader
from model import ProteinDNAGNN
from mango import  Tuner

class BaysianTuner(Trainer):
    def __init__(self, cfg):
        super(Tuner, self).__init__(cfg)
        self.cfg = cfg
        self.hyperparameters = cfg.hyperparameters
    
    def run_tuning(self, params):
        params = params[0]

        if self.config.utility.verbose.hyperparameters:
            print(f"Hyperparameters: {params}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=params["batch_size"],
            shuffle=True
        )
        self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=params["batch_size"],
                shuffle=False
            )
    
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        self.model = ProteinDNAGNN(input_dim=self.train_dataset.num_node_features, model_params=model_params, config=self.config) 
        self.model = self.model.to(self.device)

        # < 1 increases precision, > 1 recall
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=params["learning_rate"],
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=params["scheduler_gamma"])
        
        # Start training
        self.run()
        return [self.best_val_loss]

    def tune_hyperparameters(self):

        HYPERPARAMETERS = self.config_dict["hyperparameters"]
        tuner_config = self.config_dict["tuner"]

        tuner = Tuner(HYPERPARAMETERS, 
            objective=self.run_tuning,
            conf_dict=tuner_config) 
        results = tuner.minimize()

        best_params = results["best_params"]
        #Write the results to a file
        with open("../utils/tuned_params.yaml", "w") as f:
            omegaconf.OmegaConf.save(best_params, f)