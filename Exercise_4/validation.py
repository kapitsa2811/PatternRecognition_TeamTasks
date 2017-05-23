"""
Team: Over9000

Python Version: 3.6.1

Pattern Recognition - Exercise 4 - Molecules
"""

import xml.etree.ElementTree as ET
import numpy as np
import itertools
from scipy.optimize import linear_sum_assignment as lsm
import time
import os


# =============================================#
#                   Settings                   #
# =============================================#


# set the costs
Cn = 1  # cost for node deletion/insertion
Ce = 1  # cost for edge deletion/insertion

# set k-value for kNN
k = 3


# =============================================#
#               Class definitions              #
# =============================================#


class Molecule:
    """ A class for representing molecules.

    Notes
    -----
    A molecule is represented as a undirected graph. Each atom is a node and each atom-atom-connection is an edge.
    Node labels are the chemical symbols for the corresponding atom.
    In the following, the terms atom and node is used interchangeably.
    When we talk about the atom-atom-connections, we always use the term edge.

    The Molecule objects are created from a .gxl file. The file names are of the form XXX.gxl, where XXX is a
    positive integer number of length 2 to 5 without leading zeros.

    To create a molecule, the ground truth has to be given as well.

    Parameters
    ----------
    id : str
        The string contains a human readable number, representing the integer part of the filename of the molecule.
    truth : str
        Human readable string containing the ground truth of the molecules class.
        It's either 'a' for active, or 'i' for inactive.

    Attributes
    ----------
    id : int
        The integer part of the molecules filename. It's the parameter 'id' converted to int.
    atoms : list
        A list of strings, each string represents an atom of the molecule with its chemical symbol, e.g. 'P'.
    atom_num : int
        The number of atoms in the molecule, i.e. the length of the list 'atoms'.
    adj_mat : ndarray
        2D array containing information about the edges between the atoms.
        0 means no edge, 1 means edge.
    degree : ndarray
        1D array of length 'atom_num', containing positive integer values, representing the degree
        of each atom (from list 'atoms'), i.e. the number of edges connected to the atom.
        E.g. the i-th atom of list 'atoms' has degree 'degree[i]'.
    truth : int
        Containing the ground truth of the molecules class. 1 means active, 0 inactive and -1 is undefined.

    """

    def __init__(self, id, truth):
        filename = './data/gxl/' + id + '.gxl'
        self.id = int(id)
        self.atoms = get_atoms(filename)
        self.atom_num = len(self.atoms)
        self.adj_mat = get_adj_mat(filename, self.atom_num)
        self.truth = set_truth(truth)
        self.degree = np.sum(self.adj_mat, axis=0)


class Molecule2:
    """ A class for representing molecules.

    Notes
    -----
    A molecule is represented as a undirected graph. Each atom is a node and each atom-atom-connection is an edge.
    Node labels are the chemical symbols for the corresponding atom.
    In the following, the terms atom and node is used interchangeably.
    When we talk about the atom-atom-connections, we always use the term edge.

    The Molecule objects are created from a .gxl file. The file names are of the form XXX.gxl, where XXX is a
    positive integer number of length 2 to 5 without leading zeros.

    Parameters
    ----------
    id : str
        The string contains a human readable number, representing the integer part of the filename of the molecule.

    Attributes
    ----------
    id : int
        The integer part of the molecules filename. It's the parameter 'id' converted to int.
    atoms : list
        A list of strings, each string represents an atom of the molecule with its chemical symbol, e.g. 'P'.
    atom_num : int
        The number of atoms in the molecule, i.e. the length of the list 'atoms'.
    adj_mat : ndarray
        2D array containing information about the edges between the atoms.
        0 means no edge, 1 means edge.
    degree : ndarray
        1D array of length 'atom_num', containing positive integer values, representing the degree
        of each atom (from list 'atoms'), i.e. the number of edges connected to the atom.
        E.g. the i-th atom of list 'atoms' has degree 'degree[i]'.

    """

    def __init__(self, id):
        filename = './data/test/' + id + '.gxl'
        self.id = int(id)
        self.atoms = get_atoms(filename)
        self.atom_num = len(self.atoms)
        self.adj_mat = get_adj_mat(filename, self.atom_num)
        self.degree = np.sum(self.adj_mat, axis=0)


# =============================================#
#             Functions definitions            #
# =============================================#


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
        adj_mat[start - 1, end - 1] = 1
        adj_mat[end - 1, start - 1] = 1
    return adj_mat


def set_truth(truth):
    """ Return the truth value as number."""
    if truth == 'a':
        return 1
    elif truth == 'i':
        return 0
    else:
        return -1


def load_molecules(name):
    """ Return a list with all molecules listed in the file 'Exercise_4/data/name.txt'. """
    mols = list()
    with open('./data/' + name + '.txt') as f:
        for line in f:
            id, truth = str.split(line, " ")
            truth = truth.rstrip()  # remove the '\n'
            mols.append(Molecule(id, truth))
    return mols


def calc_dist(mol1, mol2, Cn, Ce):
    """ Returns the (approximate) graph edit distance of two given molecules using bipartite graph matching.

    Note
    ----
    For further elaborations see 'pr-lecture10.pdf' slides 14-21 and 'pr-lecture9.pdf' slide 36.

    The (approximate) GED is calculated by solving the assignment problem with the cost matrix C, using
    the Hungarian algorithm (from scipy.optimize).

    Let n be the number of atoms of molecule 1 and m the number of atoms of molecule 2.

    The (n+m) x (n+m) cost matrix C has four parts:
    1. Upper left: substitutions (nodes plus adjacent edges)  ->  c_i,j for i in 1..n and j in 1..m
    2. Upper right: deletions (nodes plus adjacent edges)  ->  c_i,eps for i in 1..n on the diagonal, rest are 99999
    3. Lower left: insertions (nodes plus adjacent edges)  ->  c_eps,j for j in 1..m on the diagonal, rest are 99999
    4. Lower right: dummy assignments (eps -> eps)  ->  0

    Assume u_i to be a node from molecule 1 and v_i a node from molecule 2.
    Let P_i be the set of all adjacent edges to u_i and Q_i the set of all adjacent edges to v_i.

    Then, the entries (c_i,j / c_i,eps / c_eps,j) of the cost matrix C are:
    - Deletion costs include the deletion of the node u_i as well as the deletion of all edges in P_i :
        c_i,eps = c(u_i -> eps) + sum_{p in P_i} c(p -> eps)
    - Insertion costs include the insertion of the node v_i as well as the insertion of all edges in Q_i :
        c_eps,j = c(eps -> v_j) + sum_{q in Q_i} c(eps -> q)
    - Substitution costs include the node substitution (u_i -> v_i) as well as an estimation of the
      edge assignment cost C(P_i -> Q_j) :
        c_i,j = c(u_i -> v_j) + C(P_i -> Q_j)

    For insertion and deletion cost we consider the Dirac cost function, where we fix two positive numbers Ce and Cn:
    - node substitution: c(u_i -> v_j) = 2*Cn if symbols are not equal,
                         c(u_i -> v_j) = 0    otherwise.
    - node deletion/insertion: c(u_i -> eps) = c(eps -> v_j) = Cn
    - edge deletion/insertion: c(p -> eps) = c(eps -> q) = Ce

    The estimation of the edge assignment costs C(P_i -> Q_j) are assumed to be the number of edges that have to be
    either deleted or inserted to get from P_i to Q_j weighted with the cost Ce. Hence the absolute value of the
    difference in node number of the two sets: abs(| P_i | - | Q_j |) * Ce.

    """

    # read how many atoms each molecule has
    n = mol1.atom_num
    m = mol2.atom_num

    # read how many edges each atom of each molecule has
    P = mol1.degree
    Q = mol2.degree

    # ------------------------------------------------------------------------------------------------------------------
    # COMPUTE THE ESTIMATION OF THE EDGE ASSIGNMENT COSTS C(P_i -> C_j)

    edge_ass_cost = np.array([abs(p - q) for p, q in itertools.product(P, Q)])
    edge_ass_cost = edge_ass_cost.reshape((n, m))

    # ------------------------------------------------------------------------------------------------------------------
    # COMPUTE THE COST MATRIX C

    # initiate cost matrix, filled with zeros (since the lower right part is all zeroes anyway)
    cost_mat = np.zeros((n + m, n + m))

    # compute the upper right part
    upper_right = np.diag(Cn + Ce * P)  # directly compute the diagonal matrix
    upper_right[upper_right == 0] = 99999  # replace all non-diagonal elements
    cost_mat[:n, m:] = upper_right  # insert the part in the cost matrix

    # compute the lower left part
    lower_left = np.diag(Cn + Ce * Q)  # directly compute the diagonal
    lower_left[lower_left == 0] = 99999  # replace all non-diagonal elements
    cost_mat[n:, :m] = lower_left  # insert the part in the cost matrix

    # compute the upper left part
    upper_left = [a != b for a, b in
                  itertools.product(mol1.atoms, mol2.atoms)]  # check where the symbols are not equal
    upper_left = np.array(upper_left)  # convert to np-array
    upper_left = upper_left.reshape((n, m))  # convert this True/False vector to an (nxm) matrix
    upper_left = upper_left * 2 * Cn  # convert True to 2*Cn and False to 0
    upper_left = upper_left + edge_ass_cost * Ce  # add the estimation of the edge assignment cost C(P_i -> C_j)
    cost_mat[:n, :m] = upper_left  # insert the part in the cost matrix

    # ------------------------------------------------------------------------------------------------------------------
    # COMPUTE THE GRAPH EDIT DISTANCE

    # use hungarian algorithm to find assignment with minimal cost
    row_ind, col_ind = lsm(cost_mat)

    # compute the cost of this assignment, which is the graph edit distance
    ged = cost_mat[row_ind, col_ind].sum()

    return ged


# =============================================#
#              Read the molecules              #
# =============================================#


# get the training set
train = load_molecules('train')

# get the test set
ids = [filename[:-4] for filename in os.listdir("./data/test")]
test = [Molecule2(id) for id in ids]

del ids


# =============================================#
#             Distance computation             #
# =============================================#


# compute the distance between all possible combinations of molecule pairs from test and train
print()
print('Computing the distance...')
start = time.time()
dist = np.array([calc_dist(mol1, mol2, Cn, Ce) for mol1, mol2 in itertools.product(test, train)])
dist = dist.reshape((len(test), len(train)))
end = time.time()
print('Done in {:.2f} seconds!'.format(end-start))
print()


# =============================================#
#         Save the distance matrix             #
# =============================================#


# save the computed distances
np.save('./distances/dist_test', dist)


# # =============================================#
# #        Read distance matrix from file        #
# # =============================================#
#
#
# # load the file
# dist = np.load('./distances/dist_test.npy')


# =============================================#
#                      kNN                     #
# =============================================#


# for each test-molecule, get the k indices of test-molecules with the smallest distance
idx = np.argpartition(dist, k)
idx = idx[:, 0:k]

# get the labels of those k nearest neighbors for each valid-molecule
neigh = np.array([[train[i].truth for i in row] for row in idx])

# for each valid-molecule, find the label that occurs the most (unless k=1, then we're already done)
if k == 1:
    pred = np.concatenate(neigh)
else:
    pred = np.array([np.argmax(np.bincount(row)) for row in neigh])

# convert the prediction into 'i' (0) and 'a' (1)
pred_str = ['a' if pr else 'i' for pr in pred]


file = open('output.txt', 'w')

for mol, pr in zip(test, pred_str):
    file.write(repr(mol.id) + ', ' + pr + '\n')

file.close()
