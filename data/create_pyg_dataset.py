import os
import torch
import random
import numpy as np
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split
import gemmi
from torch_cluster import radius_graph
from sklearn.model_selection import train_test_split

class ProteinDNADataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        """
        Creates a PyTorch Geometric dataset from .cif files in a given folder.
        """
        super().__init__(root=root_dir, transform=transform, pre_transform=pre_transform)
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".cif")]
        self.labels = self.load_labels()  # Assuming labels are stored in a separate file or derived

    def load_labels(self):
        """
        Load or generate binding affinity labels for the dataset.
        Example: A CSV file with {filename, binding_affinity}
        """
        labels = {}
        label_file = os.path.join(self.root_dir, "binding_affinities.csv")  # Assumed CSV file
        if os.path.exists(label_file):
            import pandas as pd
            df = pd.read_csv(label_file)
            for _, row in df.iterrows():
                labels[row["filename"]] = row["binding_affinity"]
        else:
            # Assign random labels if no file exists (for testing)
            for f in self.files:
                labels[f] = random.uniform(5.0, 10.0)  # Placeholder affinity values
        return labels

    def process_cif(self, file_path):
        """
        Parses a .cif file and converts it into a PyTorch Geometric graph.
        """
        structure = gemmi.read_structure(file_path)
        atom_features, pos = [], []

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        # Atomic features: Atomic number & 3D position
                        atom_features.append([atom.element.atomic_number])
                        pos.append(atom.pos.tolist())

        if len(pos) == 0:
            return None  # Skip empty structures

        # Convert to tensors
        x = torch.tensor(atom_features, dtype=torch.float)  # Node features
        pos = torch.tensor(pos, dtype=torch.float)  # 3D coordinates

        # Generate edges using radius graph (atoms within 5Å)
        edge_index = radius_graph(pos, r=5.0)

        # Extract label
        file_name = os.path.basename(file_path)
        y = torch.tensor([self.labels.get(file_name, 0.0)], dtype=torch.float)  # Default label if missing

        return Data(x=x, edge_index=edge_index, pos=pos, y=y)

    def len(self):
        return len(self.files)

    def get(self, idx):
        """
        Loads and processes a single .cif file as a PyTorch Geometric Data object.
        """
        file_name = self.files[idx]
        file_path = os.path.join(self.root_dir, file_name)
        return self.process_cif(file_path)

# ---- Data Preparation ----
def create_train_val_test_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the dataset into train, validation, and test sets.
    """
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Ensure sum matches total

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    return train_data, val_data, test_data

# ---- Running the Data Pipeline ----
if __name__ == "__main__":
    root_folder = "./"  # Change to your actual dataset folder

    # Load dataset
    dataset = ProteinDNADataset(root_folder)

    # Create splits
    train_dataset, val_dataset, test_dataset = create_train_val_test_splits(dataset)

    print(f"Dataset loaded! Total: {len(dataset)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Save datasets for reuse
    torch.save(train_dataset, os.path.join(root_folder, "train_dataset.pt"))
    torch.save(val_dataset, os.path.join(root_folder, "val_dataset.pt"))
    torch.save(test_dataset, os.path.join(root_folder, "test_dataset.pt"))
