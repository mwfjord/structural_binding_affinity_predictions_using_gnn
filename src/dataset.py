from attr import validate
import torch
import omegaconf
import os
from torch_geometric.data import Data, Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from complex import Complex


class PDDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None, test=False, validation=False, cfg=None):
        """
        Creates a PyTorch Geometric dataset from .cif files in a given folder.
        """
        self.root_dir = os.path.join(root_dir, "train")
        self.test = test
        self.validation = validation
        self.cfg = cfg
        self.regenerate = cfg.dataset.get("regenerate", False) if cfg is not None else False
        # self.smiles_dict = self.load_lookup("./residue_smiles.json")
        # self.unique_atoms = self.load_lookup("./uniqueAtoms.json")
        

        if self.regenerate:
            print("Regenerating dataset...")
        if test:
            self.root_dir = os.path.join(root_dir, "test")
        if validation:
            self.root_dir = os.path.join(root_dir, "validation")
        if test and validation:
            raise ValueError("Cannot have both test and validation set at the same time.")
        
        
        raw_dir = os.path.join(self.root_dir, "raw")
        processed_dir = os.path.join(self.root_dir, "processed")
        self.files = [f for f in os.listdir(raw_dir) if f.endswith(".cif")]
        self.labels = self.load_labels()
        
        super(PDDataset, self).__init__(self.root_dir, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return self.files
    
    @property
    def processed_file_names(self):
        # return 'some_temoprary_file_name' # This is a temporary placeholder so that i can tinker with the data processing part
        if not self.regenerate:
            if self.test:
                return [f'data_test_{i}.pt' for i in range(len(self.files))]
            else:
                return [f'data_{i}.pt' for i in range(len(self.files))]
        else:
            return ' '
    
    def download(self):
        pass
    
    
    
    def _get_label(self, file_name):

        """
        Get the binding affinity label for a given file.
        """
        return self.labels[file_name]
        
    def load_labels(self):
        """
        Load or generate binding affinity labels for the dataset.
        Example: A CSV file with {origSequence_name, binding_affinity}
        """
        labels = {}
        label_file = os.path.join( self.root_dir, "binding_affinities.csv")
        
        if os.path.exists(label_file):
            df = pd.read_csv(label_file, delimiter='\t')
            for col, row in df.iterrows():
                log_K_d = np.log(row["KD"], dtype=np.float32)
                log_K_d = torch.tensor(log_K_d, dtype=torch.float32)
                labels[row["origSequence_name"]] = log_K_d
        else:
            print("No label file found. Generating random labels...")
        return labels


        
    def process(self):
        """
        Parses a .cif file and converts it into a PyTorch Geometric graph.
        """

        print("Processing data...")
        for index, file in tqdm(enumerate(self.files)):
            sequence = file.split('.')[0]
            file_path = os.path.join(self.root_dir, "raw", file)
            
            pdb = Complex(file_path)
            edge_index = pdb._get_edge_index()
            edge_features = pdb._get_edge_attr()
            node_features = pdb._get_node_features()
            label = self._get_label(sequence)
            
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=label)
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))

    def len(self):
        return len(self.files)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'), weights_only=False)
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'), weights_only=False)   
        return data
    


# cfg = omegaconf.OmegaConf.load("../utils/config.yaml")
# data_set = PDDataset(root_dir="../data", test=False, validation=False, cfg=cfg)