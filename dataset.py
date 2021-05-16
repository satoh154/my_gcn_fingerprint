import pandas as pd
import numpy as np
from torch.utils import data


class SmilesDataLoader:

    def __init__(self, csv_path:str, smiles_col:str="smiles", label_props:str=["label"]):
        df = pd.read_csv(csv_path)

        mols = []
        labels_list = []
        for i, samples in enumerate(df[smiles_col].values):
            from rdkit import Chem
            mol = Chem.MolFromSmiles(samples)
            if mol is not None:
                mols.append(mol)
                labels = [None] * len(label_props)
                for j, label in enumerate(label_props):
                    labels[j] = df[label].values[i]

                labels_list.append(labels)

        self.mols = mols
        self.labels_list = labels_list


class MoleculeDataset(data.Dataset):

    def __init__(self, mol_list, label_list):
        self.mol_list = mol_list
        self.label_list = label_list

    def __len__(self):
        return len(self.mol_list)

    def __getitem__(self, index):
        return self.mol_list[index], self.label_list[index]

