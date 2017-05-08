"""
Team: Over9000

Python Version: 3.6

Pattern Recognition - Exercise 4
"""

import xml.etree.ElementTree as ET
import numpy as np


class Molecule:

    def __init__(self, id, truth):
        filename = './data/gxl/' + id + '.gxl'
        self.id = int(id)
        self.atoms = get_atoms(filename)
        self.length = len(self.atoms)
        self.adj_mat = get_adj_mat(filename, self.length)
        self.truth = truth

    def set_truth(self, str):
        self.truth = str


def get_atoms(filename):
    tree = ET.parse(filename)
    atoms = list()
    for node in tree.findall(".//node/attr/string"):
        node_value = node.text
        atoms.append(node_value.strip())
    return atoms


def get_adj_mat(filename, n):
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

    mols = list()

    with open('./data/' + name + '.txt') as f:
        for line in f:
            id, truth = str.split(line, " ")
            mols.append(Molecule(id, truth))
    return mols

train = load_molecules('train')
valid = load_molecules('valid')
