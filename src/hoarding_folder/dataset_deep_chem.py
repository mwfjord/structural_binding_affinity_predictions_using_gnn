

from ast import Tuple
from html import parser
import gemmi
from networkx import moral_graph
from sklearn import neighbors
import torch
import os
from torch_geometric.data import Data, Dataset
import json
from tqdm import tqdm
import numpy as np
import rdkit as rd
import deepchem as dc
from deepchem.feat import AtomicConvFeaturizer, RdkitGridFeaturizer, MolGraphConvFeaturizer
import omegaconf
import pandas as pd

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import MMCIFParser, PDBIO
import tempfile
from gemmi import cif
import sire as sr
from rdkit import Chem
import petidy as pt
import 



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
        print("Dataset created.")
        
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

    def get_rdkit_molecules(self, file_name):
        # Create a temporary directory
        parser = MMCIFParser()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        structure = parser.get_structure("complex", file_name)
        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save("output.pdb")
        mol = Chem.MolFromPDBFile("output.pdb", removeHs=False)
        return mol
        


    def create_graph_features(self, file_path):
        """
        Creates node features, edge features and edge index for a given .cif file.
        """
        pass


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
        # featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        # Parse the mmCIF file.
        # Define maximum atom numbers based on your system (example numbers)
        frag1_num_atoms = 500   # maximum atoms in DNA
        frag2_num_atoms = 2000  # maximum atoms in protein
        complex_num_atoms = frag1_num_atoms + frag2_num_atoms
        max_num_neighbors = 1200
        neighbor_cutoff = 45  # in angstroms

        # Initialize the featurizer (strip_hydrogens as desired)
        # featurizer = AtomicConvFeaturizer(
        #     frag1_num_atoms,
        #     frag2_num_atoms,
        #     complex_num_atoms,
        #     max_num_neighbors,
        #     neighbor_cutoff,
        #     strip_hydrogens=True
        # )
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        parser = MMCIFParser()
        complexes = []
        for idx, file in enumerate(tqdm(self.files)):
            print(f"Processing file {file}")
            file_id = file.split('.')[0]
            structure = parser.get_structure("complex", os.path.join(self.root_dir, "raw", file))
            io = PDBIO()
            io.set_structure(structure)
            io.save(f"complex_{file_id}.pdb")
            # Save the protein and DNA structures to separate PDB files with file index

            # io.save(f"protein_{file_id}.pdb", ProteinSelect())
            # io.set_structure(structure)
            # io.save(f"dna_{file_id}.pdb", DNASelect())
            try:
                rdkit_complex = Chem.MolFromPDBFile(f"complex_{file_id}.pdb", removeHs=False)
            except:
                print(f"Error in processing file to rdkit {file}")
                continue
            try:
                f = featurizer.featurize(rdkit_complex)
            except:
                print(f"Error in featurizing file with DeepChem {file}")
                continue
            try:
                data = f[0].to_pyg_graph()
            except:
                print(f"Error in converting to PyG graph {file}")
                continue

            complexes.append(data)
            print(data)
        print(complexes)




        # print(complexes)
        # print("Processing data...")
        # for index, file in tqdm(enumerate(self.files)):
        #     sequence = file.split('.')[0]
        #     file_path = os.path.join(self.root_dir, "raw", file)
        #     mol_graphs = self.get_rdkit_molecules(file_path)
        #     if mol_graphs is None:
        #         print(f"Error in processing file {file}")
        #         continue
        #     f = featurizer.featurize(mol_graphs)
        #     data = f[0].to_pyg_graph()
        #     label = self._get_label(sequence)
        #     data.y = label
        #     # data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=label)
        #     # validate_graph(data)
        #     if self.test:
        #         torch.save(data, 
        #             os.path.join(self.processed_dir, 
        #                         f'data_test_{index}.pt'))
        #     else:
        #         torch.save(data, 
        #             os.path.join(self.processed_dir, 
        #                         f'data_{index}.pt'))


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


cfg = omegaconf.OmegaConf.load("config.yaml")
data_set = PDDataset(root_dir="../data", test=False, validation=False, cfg=cfg)