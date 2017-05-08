"""
Team: Over9000

Python Version: 3.6

Pattern Recognition - Exercise 4
"""

import xml.etree.ElementTree as ET
import numpy as np


class Molecule:
    """ A class for representing molecules.

    Parameters
    ----------
    id : str
        The string contains a human readable number, representing the filename of the molecule.
     truth : str
        Human readable string containing the ground truth of the molecules class.
        It's either 'a' for active, or 'i' for inactive.

     Attributes
     ----------
     id : int
        The number contained in the molecules filename, extracted from parameter 'id'.
     atoms : list
        A list of strings, each string represents an atom with its chemical symbol.
     length : int
        The number of atoms, i.e. the length of tha atoms list.
     adj_mat : ndarray
        2D array containing the covalence bonds between the atoms stored in the atoms list.
        0 means no bond, 1 means single linkage and 2 means double linkage.
     truth : str
        Human readable string containing the ground truth of the molecules class.
        Either 'a' for active or 'i' for inactive.

    """

    def __init__(self, id, truth):
        filename = './data/gxl/' + id + '.gxl'
        self.id = int(id)
        self.atoms = get_atoms(filename)
        self.length = len(self.atoms)
        self.adj_mat = get_adj_mat(filename, self.length)
        self.truth = truth


def get_atoms(filename):
    """ Return a list of all atoms from the molecule stored in 'filename'. """
    tree = ET.parse(filename)
    atoms = list()
    for node in tree.findall(".//node/attr/string"):
        node_value = node.text
        atoms.append(node_value.strip())
    return atoms


def get_adj_mat(filename, n):
    """ Return the adjacency matrix from the molecule stored in 'filename'. """
    tree = ET.parse(filename)
    adj_mat = np.zeros((n, n), dtype=int)
    for edge in tree.findall(".//edge"):
        start = int(edge.get('from')[1:])
        end = int(edge.get('to')[1:])
        val = int(edge[0][0].text)
        adj_mat[start-1, end-1] = val
        adj_mat[end-1, start-1] = val
    return adj_mat


def load_molecules(name):
    """ Return a list with all molecules listed in the file Exercies_4/data/name.txt"""
    mols = list()
    with open('./data/' + name + '.txt') as f:
        for line in f:
            id, truth = str.split(line, " ")
            mols.append(Molecule(id, truth))
    return mols

# Get the training- and validation sets

train = load_molecules('train')
valid = load_molecules('valid')
