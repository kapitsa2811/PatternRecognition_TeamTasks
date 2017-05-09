"""
Team: Over9000

Python Version: 3.6

Pattern Recognition - Exercise 4
"""

import xml.etree.ElementTree as ET
import numpy as np
import itertools


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
     degree : ndarray
        1D array of length 'length', containing positive integer values, representing the degree
        of each atom (from list 'atoms'), i.e. number of edges connected to the node (atom).
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
        self.degree = np.sum(self.adj_mat, axis=0)


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
    """ Return a list with all molecules listed in the file 'Exercise_4/data/name.txt'. """
    mols = list()
    with open('./data/' + name + '.txt') as f:
        for line in f:
            id, truth = str.split(line, " ")
            mols.append(Molecule(id, truth))
    return mols

# Get the training- and validation sets

train = load_molecules('train')
valid = load_molecules('valid')

def calc_cost_matrix(mol1, mol2):
    """ Returns the cost matrix for two given molecules."""

    # read how many atoms each molecule has
    n = mol1.length
    m = mol2.length

    # set the costs
    Cn = 1  # cost for node deletion/insertion:
    Ce = 1  # cost for edge deletion/insertion

    # initiate cost matrix, filled with zeros
    cost_mat = np.zeros((n+m, n+m), dtype=float)

    # enter deletion for mol1
    upper_right = np.diag(Cn + Ce*mol1.degree)
    upper_right[upper_right == 0] = -1
    cost_mat[:n, m:] = upper_right

    # enter insertions for mol2
    lower_left = np.diag(Cn + Ce*mol2.degree)
    lower_left[lower_left == 0] = -1
    cost_mat[n:, :m] = lower_left

    # enter substitutions
    mat = np.array([a!=b for a, b in itertools.product(mol1.atoms, mol2.atoms)])
    mat = mat.reshape((n, m))
    mat = mat*2*Cn
    cost_mat[:n, :m] = mat

    return cost_mat