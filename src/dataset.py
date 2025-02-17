import dis
from platform import node
from turtle import distance
from matplotlib.pyplot import bone
import torch
from htmd.ui import *
from moleculekit.bondguesser import *
import omegaconf
import os
from torch_geometric.data import Data, Dataset
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
# from torch_geometric.utils import radius_graph








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
        self.unique_atoms = self.load_lookup("./uniqueAtoms.json")
        

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
                # print(log_K_d)
                labels[row["origSequence_name"]] = log_K_d
        else:
            print("No label file found. Generating random labels...")
        return labels

    def create_graph_features(self, file_path):
        """
        Creates node features, edge features and edge index for a given .cif file.
        """
        mol = Molecule(file_path)
        # mol = systemPrepare(mol)
        # mol = autoSegment(mol)
        # mol = solvate(mol)
        Molecule.guessBonds(mol)
        # mol.wrap()
        node_features = torch.tensor([], dtype=torch.float32)
        edge_features = torch.tensor([], dtype=torch.float32)        
        # Neighbor list (2, num_edges)
        edge_index = torch.tensor([], dtype=torch.long)
        bonds_count = 0
        for i in range(mol.numAtoms):
            atom_encoding = self.atom_to_onehot(mol.element[i])
            atom_charge = mol.charge[i]
            atom_pos = mol.coords[i]
            for bond in mol.bonds[i]:
                bonds_count += 1
                edge_index = torch.cat((edge_index, torch.tensor([[i, bond]], dtype=torch.long).T), dim=1)
                distance = np.linalg.norm(atom_pos - mol.coords[bond])
                distance = 1/(1+(distance/10))
                edge_features = torch.cat((edge_features, torch.tensor([distance], dtype=torch.float32)))


            node_features = torch.cat((node_features, torch.tensor([atom_encoding + [atom_charge] + list(atom_pos)], dtype=torch.float32)))

        if self.cfg.utility.verbose:
            print(f"\nNumber of bonds: {bonds_count}")
            print("Data dimensions: ", node_features.shape, edge_index.shape, edge_features.shape)

        return node_features, edge_index, edge_features




    def atom_to_onehot(self, atom_name):
        # Convert residue names to one-hot vectors for amino acids and DNA bases
        if atom_name in self.unique_atoms:
            index = self.unique_atoms.index(atom_name)
            onehot = [0] * len(self.unique_atoms)
            onehot[index] = 1
            return onehot
        else:
            return [0] * len(self.unique_atoms)
    
    def process(self):
        """
        Parses a .cif file and converts it into a PyTorch Geometric graph.
        """

        print("Processing data...")
        for index, file in tqdm(enumerate(self.files)):
            sequence = file.split('.')[0]
            file_path = os.path.join(self.root_dir, "raw", file)
            node_features, edge_index, edge_features = self.create_graph_features(file_path)
            label = self._get_label(sequence)
            # edge_features = graph_data["edge_features"]
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=label)
            # validate_graph(data)
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
    
    def load_lookup(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)


# cfg = omegaconf.OmegaConf.load("config.yaml")
# data_set = PDDataset(root_dir="../data", test=False, validation=False, cfg=cfg)