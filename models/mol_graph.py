from rdkit import Chem
import numpy as np

from models.util import atom_features, bond_features


class MolGraph(object):

    def __init__(self):
        self.nodes = {}
        self.degrees = [0, 1, 2, 3, 4, 5]

    def new_node(self, ntype, features=None, rdkit_idx=None, labels=None):
        new_node = Node(ntype, features, rdkit_idx, labels=labels)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in self.degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in self.degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_idx_array(self):
        return np.array([node.rdkit_idx for node in self.nodes['atom']])

    def labels_array(self):
        return np.array([node.labels for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]


class Node(object):

    def __init__(self, ntype, features, rdkit_idx, labels=None):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_idx = rdkit_idx
        self.labels = labels

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]


def graph_from_mol_tuple(mol_tuple, atom_labels_list=None, atom_features_list=None):
    graph_list = []
    for i, m in enumerate(mol_tuple):
        atom_labels = None
        atom_features = None
        if atom_labels_list:
            atom_labels = atom_labels_list[i]
        if atom_features_list:
            atom_features = atom_features_list[i]
        graph_list.append(graph_from_mol(m, atom_labels, atom_features))

    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)

    big_graph.sort_nodes_by_degree('atom')
    return big_graph


def graph_from_mol(mol, atom_labels=None, atom_features_exp=None):
    graph = MolGraph()
    atoms_by_rd_idx = {}

    for atom in mol.GetAtoms():
        features = atom_features(atom)
        if atom_features_exp:
            features = np.hstack([features, atom_features_exp[atom.GetIdx()]])

        if atom_labels:
            new_atom_node = graph.new_node('atom', labels=atom_labels[atom.GetIdx()], features=features,
                                           rdkit_idx=atom.GetIdx())
        else:
            new_atom_node = graph.new_node('atom', features=features, rdkit_idx=atom.GetIdx())

        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph


def graph_from_smiles_tuple(smiles_tuple):
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)

    big_graph.sort_nodes_by_degree('atom')
    return big_graph


def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_idx=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph


def gcn_collate_fn(batch):

    mols = []
    labels = []
    atom_labels = []
    atom_features = []

    for i, (mol, label, *atom_data) in enumerate(batch):
        mols.append(mol)
        labels.append(label)
        if len(atom_data) > 0:
            atom_labels.append(atom_data[0])
            if len(atom_data) > 1:
                atom_features.append(atom_data[1])

    molgraph = graph_from_mol_tuple(mols, atom_labels, atom_features)
    arrayrep = {
            'atom_features': molgraph.feature_array('atom'),
            'bond_features': molgraph.feature_array('bond'),
            'atom_list': molgraph.neighbor_list('molecule', 'atom'), 
            'rdkit_ix': molgraph.rdkit_idx_array(),
            'atom_labels': molgraph.labels_array()
            }

    for degree in molgraph.degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)

    return arrayrep, labels

