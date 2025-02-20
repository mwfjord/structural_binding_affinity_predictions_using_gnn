from math import dist
import torch
import numpy as np
from rdkit import Chem
from Bio.PDB import MMCIFParser, PDBIO, Select
import json
import subprocess
import sys
from rdkit.Chem import AllChem




with open("../utils/residue_smiles.json") as f:
    residue_smiles = json.load(f)

# with open("./uniqueAtoms.json") as f:
#     unique_atoms = json.load(f)


# Define selection classes for protein and DNA.
class ProteinSelect(Select):
    def accept_residue(self, residue):
        # List of standard amino acids
        protein_residues = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
            'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
            'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        }
        return residue.get_resname().upper() in protein_residues

class DNASelect(Select):
    def accept_residue(self, residue):
        # Typical nucleotide residue names in mmCIF files
        dna_residues = {'DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'T'}
        return residue.get_resname().upper() in dna_residues
    



class Residue():
    """
    Class to represent a residue in a protein-DNA complex
    """
    def __init__(self, residue):
        self.residue = residue
        self.id = residue.get_id()
        self.protein_or_dna = self._get_protein_or_dna()
        self.resname = residue.get_resname()
        self.smiles = self._get_smiles()
        self.atoms = self._get_atoms()
        self.rdkit_mol = self._get_rdkit_mol()
        self.molecular_fingerprint = self._get_molecular_fingerprint()
        self.center = self._get_center()
        self.features = self._merge_features()

    def _get_features(self):
        """
        Get the features of the residue
        """
        return np.array([self.protein_or_dna] + list(self.center) + list(self.molecular_fingerprint))

    def _get_atoms(self):
        """
        Get the atoms in the residue
        """
        return [atom for atom in self.residue.get_atoms()]  
    
    def _get_smiles(self):
        """
        Get the SMILES representation of the residue
        """
        return residue_smiles.get(self.resname, None)
    
    def _get_rdkit_mol(self):
        """
        Get the RDKit molecule representation of the residue
        """
        if self.smiles:
            return Chem.MolFromSmiles(self.smiles)
        return None
    
    def _get_molecular_fingerprint(self):
        """
        Get the molecular fingerprint of the residue
        """
        fpgen = AllChem.GetRDKitFPGenerator()
        ao = AllChem.AdditionalOutput()
        ao.CollectBitPaths()
        fp = fpgen.GetSparseCountFingerprint(self.rdkit_mol ,additionalOutput=ao)
        if fp is not None:
            return fp
        return None
    
    def _get_center(self):
        """
        Get the center of the residue
        """
        center = np.array([0.0, 0.0, 0.0])
        for atom in self.atoms:
            center += atom.get_coord()
        return center / len(self.atoms)
    
    def _get_protein_or_dna(self):
        """
        Get the type of the residue (protein or DNA)
        """
        if ProteinSelect().accept_residue(self.residue):
            return 1
        elif DNASelect().accept_residue(self.residue):
            return 0
        else:
            raise ValueError("Unknown residue type")
    
    def _merge_features(self):
        """
        Get the features of the residue
        """
        return np.array([self.protein_or_dna] + list(self.center) + list(self.molecular_fingerprint))

class Complex():
    """
    Class to parse and represent protein-DNA complexes from strutural files
    """
    def __init__(self, file):
        self.cif_file = file
        self.structure = None
        self.file = self._convert_file2pdb()
        self.edge_index, self.edge_attr = self._get_contacts()
        self.residues = []
        self.bond_types = ['hp', 'sb', 'pc', 'ps', 'ts', 'vdw', 'hb']

    def _get_edge_index(self):
        return torch.tensor(self.edge_index, dtype=torch.long)
    
    def _get_edge_attr(self):
        return torch.tensor(self.edge_attr, dtype=torch.float32)
    
    def _get_node_features(self):
        return torch.tensor([residue._get_features for residue in self.residues], dtype=torch.float32)

    def bond_types2onehot(self, bond):
        """
        Convert a list of bond types to one-hot encoding
        """
        onehot = np.zeros(len(self.bond_types))
        onehot[self.bond_types.index(bond)] = 1
        return onehot
    
    def get_distance(self, residue_index, nucleotide_index):
        """
        Get the distance between a residue and a nucleotide
        """
        residue = self.residues[residue_index]
        nucleotide = self.residues[nucleotide_index]
        residue = [residue.get_center()]
        nucleotide = [nucleotide.get_center()]
        return dist(residue, nucleotide)

    def _convert_file2pdb(self):
        """
        Parse the complex structure from the PDB file
        """
        parser = MMCIFParser()
        self.structure = parser.get_structure("complex", self.cif_file)
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    res = Residue(residue)
                    self.residues.append(res)
        io = PDBIO()
        io.set_structure(self.structure)
        file_path = "../data/temp/complex.pdb"
        io.save(file_path)
        return file_path
    
    def _get_contacts(self):
        # Define the path to the script and the arguments
        script_path = "../getcontacts/get_static_contacts.py"

        args = ["--structure", self.file, 
                "--sele", "nucleic", 
                "--sele2", "protein", 
                "--itypes", "all", 
                "--output", "../data/temp/contacts.tsv", 
                "--ps_cutoff_dist", "100", 
                "--hbond_cutoff_ang", "70"]

        # Construct the command: use "python" (or "python3" if needed) and pass the script and its arguments
        command = [sys.executable, script_path] + args

        # Run the command and capture the output
        subprocess.run(command, capture_output=True, text=True)
        
        # Parse the tsv file to get the contacts
        edge_index, edge_attr =  self.parse_chemical_connections("../data/temp/contacts.tsv")
        return edge_index, edge_attr
    
    def parse_chemical_connections(self, filename):
        # Dictionary to store the parsed connections.
        # For example, keys are residues and values are lists of connections.
        edge_index = [] # Shape: [2, num_edges]
        edge_attr = []  # Shape: [num_edges, num_bond_types + 1]
        
        with open(filename, 'r') as file:
            for line in file:
                # Remove leading/trailing whitespace
                line = line.strip()
                
                # Skip empty lines or lines that start with a comment symbol
                if not line or line.startswith('#'):
                    continue
                
                # Split the line into parts.
                # Modify the delimiter (e.g., comma, tab, whitespace) as needed.
                parts = line.split('\t')  # or line.split(',') if CSV
                
                # Make sure there are at least two parts: residue and nucleotide.
                if len(parts) < 2:
                    continue
                
                bond_type = self.bond_types2onehot(parts[1])
                residue = parts[2]
                residue = residue.split(":")[1]
                residue_index = residue[2]
                nucleotide = parts[3]
                nucleotide = nucleotide.split(":")[1]
                nucleotide_index = nucleotide[2]
                distance = self.get_distance(residue, nucleotide)
                # Add the connection to the dictionary.
                edge_index.append([residue_index, nucleotide_index])
                edge_attr.append(bond_type + [distance])

        return edge_index, edge_attr

data = Complex("../data/example.cif")
print(data.contacts)