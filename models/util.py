from rdkit import Chem
import numpy as np


def encoding_onehot(x, iterable):
    if x not in iterable:
        raise Exception("input {0} not in allowable set{1}:".format(x, iterable))
    return list(map(lambda s: x == s, iterable))


def encoding_onehot_unk(x, iterable):
    if x not in iterable:
        x = iterable[-1]
    return list(map(lambda s: x == s, iterable))


def atom_features(atom:Chem.Atom):
    return np.array(encoding_onehot_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    encoding_onehot(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    encoding_onehot_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    encoding_onehot_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])


def bond_features(bond:Chem.Bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])
