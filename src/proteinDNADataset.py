import os
import torch
import random
import numpy as np
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split
import gemmi
from torch_cluster import radius_graph
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

class ProteinDNADataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None, test=False, validation=False):
        """
        Creates a PyTorch Geometric dataset from .cif files in a given folder.
        """
        self.root_dir = os.path.join(root_dir, "train")
        self.test = test
        self.validation = validation
        
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
        super(ProteinDNADataset, self).__init__(self.root_dir, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return self.files
    
    @property
    def processed_file_names(self):
        # return 'some_temoprary_file_name' # This is a temporary placeholder so that i can tinker with the data processing part
        if self.test:
            return [f'data_test_{i}.pt' for i in range(len(self.files))]
        else:
            return [f'data_{i}.pt' for i in range(len(self.files))]
    
    def download(self):
        pass
    
    def _get_node_features(self, structure):
        """
        Extracts node features from a structure.
        """
        atom_features, pos = [], []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        # Atomic features: Atomic number & 3D position
                        atom_features.append([atom.serial])
                        pos.append(atom.pos.tolist())
                        
                        
        return torch.tensor(atom_features, dtype=torch.float), torch.tensor(pos, dtype=torch.float)
    
    def _get_edge_features(self, pos):
        """
        Generates edges features 
        """
        pass
    
    def _adjacency_info(self, pos):
        """
        Generates edges using radius graph.
        """
        edge_index = radius_graph(pos, r=5.0, batch=None, loop=True)
        return edge_index
    
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
                labels[row["origSequence_name"]] = row["KD"]
        else:
            print("No label file found. Generating random labels...")
            
        return labels

    def process(self):
        
        """
        Parses a .cif file and converts it into a PyTorch Geometric graph.
        """
        
        for index, file in tqdm(enumerate(self.files)):
            sequence = file.split('.')[0]
            structure = gemmi.read_structure(os.path.join(self.root_dir, "raw", file))
            atom_type, pos = self._get_node_features(structure)
            atom_features = torch.cat([atom_type, pos], dim=1)
            edge_index = self._adjacency_info(pos)
            label = self._get_label(sequence)
            data = Data(x=atom_features, edge_index=edge_index, y=label)
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
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data

