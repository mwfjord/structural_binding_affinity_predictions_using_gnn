import os
import torch
import random
import json
import numpy as np
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split
# from rdkit.Chem import rdFingerprintGenerator, MolFromPDBBlock, MolFromPDBFile
# from Bio.PDB import *
from Bio.PDB import NeighborSearch, Polypeptide
# from torch_cluster import radius_graph
from rdkit import Chem
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
import pandas as pd
from tqdm import tqdm

from Bio.PDB import MMCIFParser
import time

def get_hydrophobicity(atom):
    # Example hydrophobicity table
    hydrophobicity_table = {
        'C': 0.5, 'H': 0.1, 'O': -0.7, 'N': -0.5, 'P': -0.3, 'S': 0.2
    }
    return hydrophobicity_table.get(atom.element, 0.0)

def residue_to_onehot(residue_name):
    # Convert residue names to one-hot vectors for amino acids and DNA bases
    residue_types = {}
    with open("./residue_smiles.json", "r") as f:
            residue_types = json.load(f).keys()
    return [1 if residue_name == res else 0 for res in residue_types]




def generate_edge_features(atoms, cutoff=5.0):
    coords = np.array([atom.coord for atom in atoms])
    kdtree = cKDTree(coords)
    pairs = kdtree.query_pairs(r=cutoff)  # Finds all pairs within the cutoff radius
    
    # Create edge indices
    edge_indices = torch.tensor(list(pairs), dtype=torch.long).t()  # Shape: (2, num_edges)
    
    # Compute edge features (distances)
    edge_features = torch.norm(
        torch.tensor(coords[edge_indices[0]] - coords[edge_indices[1]], dtype=torch.float32),
        dim=1
    ).unsqueeze(-1)  # Shape: (num_edges, 1)
    
    return edge_indices, edge_features



def generate_node_features(atoms, cutoff=5.0):
    coords = torch.tensor([atom.coord for atom in atoms], dtype=torch.float32)  # Shape: (N, 3)
    atom_types = [atom.element for atom in atoms]
    
    # Map atom types to numeric indices
    unique_atom_types = list(set(atom_types))
    atom_type_map = {element: idx for idx, element in enumerate(unique_atom_types)}
    atom_type_tensor = torch.tensor([atom_type_map[atom] for atom in atom_types], dtype=torch.float32)

    # Use radius_graph for neighbors
    edge_index = radius_graph(coords, r=cutoff)
    neighbor_counts = torch.bincount(edge_index[0], minlength=len(atoms)).float()
    
    # Compute minimum distances for each atom
    distances = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], dim=1)
    min_distances = torch.zeros(len(atoms), dtype=torch.float32).scatter_add_(
        0, edge_index[0], distances
    )

    # Combine features into a single tensor
    node_features = torch.cat([
        atom_type_tensor.unsqueeze(1),  # Shape: (N, 1)
        coords,                         # Shape: (N, 3)
        neighbor_counts.unsqueeze(1),   # Shape: (N, 1)
        min_distances.unsqueeze(1)      # Shape: (N, 1)
    ], dim=1)                           # Final shape: (N, 6)
    
    return node_features



def create_graph_features(file_path, cutoff=5.0):
    verbose = False
    timers = {}
    start_time = time.time()
    # Start timer
    structure, protein_chains, dna_chains, atoms, ns = parse_cif_create_features(file_path)
    end_time = time.time()
    parse_time = end_time - start_time
    
    node_features = generate_node_features(atoms, cutoff)
    end_time = time.time()
    node_time = end_time - parse_time

    edges, edge_features = generate_edge_features(atoms, cutoff)
    end_time = time.time()
    edge_time = end_time - node_time
    
    total_time = end_time - start_time
    
    if verbose:
        print(f"Parse time: {parse_time:.2f} s")
        print(f"Node feature generation time: {node_time:.2f} s")
        print(f"Edge feature generation time: {edge_time:.2f} s")
        print(f"Total time: {total_time:.2f} s")
    
    return {
        "node_features": node_features,
        "edges": edges,
        "edge_features": edge_features,
    }


def parse_cif_create_features(file_path):
    # Parse the mmCIF file
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("complex", file_path)
    
    protein_chains = []
    dna_chains = []

    for chain in structure.get_chains():
        # Check if the chain contains amino acids (protein) or DNA residues
        if all(Polypeptide.is_aa(residue, standard=True) for residue in chain.get_residues()):
            protein_chains.append(chain)
        elif all(residue.get_resname() in ['DA', 'DC', 'DG', 'DT'] for residue in chain.get_residues()):
            dna_chains.append(chain)
    
    atoms = list(structure.get_atoms())
    ns = NeighborSearch(atoms)  # Neighbor search object for distance-based features

    return structure, protein_chains, dna_chains, atoms, ns

def validate_graph(data, num_node_features=6, num_edge_features=1):
    # Validate x (Node Features)
    if data.x.shape[1] != num_node_features:
        raise ValueError(f"Inconsistent node feature dimension: {data.x.shape[1]} != {num_node_features}")
    
    # Validate edge_index
    if data.edge_index.shape[0] != 2:
        raise ValueError(f"Inconsistent edge_index shape: {data.edge_index.shape[0]} != 2")
    
    # Validate edge_attr
    if data.edge_attr.shape[1] != num_edge_features:
        raise ValueError(f"Inconsistent edge feature dimension: {data.edge_attr.shape[1]} != {num_edge_features}")
    
    # Validate y (Labels)
    if not isinstance(data.y, torch.Tensor) or data.y.numel() == 0:
        raise ValueError("Missing or invalid label (y)")




class ProteinDNADataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None, test=False, validation=False, cfg=None):
        """
        Creates a PyTorch Geometric dataset from .cif files in a given folder.
        """
        self.root_dir = os.path.join(root_dir, "train")
        self.test = test
        self.validation = validation
        self.cfg = cfg
        self.regenerate = cfg.dataset.get("regenerate", False) if cfg is not None else False
        self.smiles_dict = self.load_smiles_lookup("./residue_smiles.json")
        
        # print("TEST")
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
        super(ProteinDNADataset, self).__init__(self.root_dir, transform, pre_transform)
        
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
    
    

    
    def _adjacency_info(self, pos):
        """
        Generates edges using radius graph.
        """
        edge_inclusion_radius = self.cfg.dataset.get("edge_inclusion_radius") # if self.cfg is not None else 30
        edge_index = radius_graph(pos, r=edge_inclusion_radius)
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
                log_K_d = np.log(row["KD"], dtype=np.float32)
                log_K_d = torch.tensor(log_K_d, dtype=torch.float32)
                # print(log_K_d)
                labels[row["origSequence_name"]] = log_K_d
        else:
            print("No label file found. Generating random labels...")
        return labels

    def process(self):
        
        """
        Parses a .cif file and converts it into a PyTorch Geometric graph.
        """
        
        

        for index, file in tqdm(enumerate(self.files)):
            sequence = file.split('.')[0]
            file_path = os.path.join(self.root_dir, "raw", file)
            graph_data = create_graph_features(file_path)
            label = self._get_label(sequence)
            atom_features = graph_data["node_features"]
            edge_index = graph_data["edges"]
            edge_features = graph_data["edge_features"]
            data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_features, y=label)
            validate_graph(data)
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
    


    def load_smiles_lookup(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)





