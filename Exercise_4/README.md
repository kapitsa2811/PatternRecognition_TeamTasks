# Exercise 4 - Molecules


## Packages
- xml.etree.ElementTree
- numpy 1.12.0

## Data
The folder `Exercise_04/data/gxl` contains 500 [gxl-files](https://en.wikipedia.org/wiki/GXL), each representing a graph of a molecular compound. Each molecule has either activity against HIV or not. So we consider two classes, active (a) and inactive (i). The files are named with a unique number, which we consider as an identification number (id).

The files `Exercise_04/data/train.txt`and `Exercise_04/data/valid.txt` contain the partitioning into training and validation set (each contains 250 molecules), as well as the ground-truth, i.e. if the molecule belongs to the class 'active' or 'inactive'. In both files, each line represents one molecule in two columns (separated by a simple space). The first column states the identification number of the molecule and the second column contains either the letter 'a' or 'i', representing the  class the molecule belongs to.

The gxl-files are structured as follows

```xml
<?xml version="1.0"?>
<!DOCTYPE gxl SYSTEM "http://www.gupro.de/GXL/gxl-1.0.dtd"><gxl>
  <graph id="molid256" edgeids="false" edgemode="undirected">
    <node id="_1">
      <attr name="symbol"><string>P  </string></attr>
      <attr name="chem"><int>6</int></attr>
      <attr name="charge"><int>0</int></attr>
      <attr name="x"><float>3</float></attr>
      <attr name="y"><float>0.25</float></attr>
    </node>
    <node id="_2">
      <attr name="symbol"><string>O  </string></attr>
      <attr name="chem"><int>2</int></attr>
      <attr name="charge"><int>0</int></attr>
      <attr name="x"><float>3</float></attr>
      <attr name="y"><float>1.25</float></attr>
    </node>
    <edge from="_1" to="_2">
      <attr name="valence"><int>1</int></attr>
    </edge>
  </graph>
</gxl>
```
this snippet represents the (nonsense) molecule with two atoms, P and O, bonded with valence 1. So a molecule is represented as a graph: atoms are edges, where their labels are their chemical symbol, and the covalent bonds between atoms are edges, where their labels are the valence of their linkage (simple or double).


## Description

In this exercise, the molecules of the validation set are classified using KNN with the approximate graph edit distance (GED). The validation set and the training set contain 250 molecules each.

In a first step, the provided dataset is converted into an appropriate data structure. Each molecule is represented as an object of class `Molecule` with the following attributes
- `id` : the integer part of the molecules filename
- `atoms` : a (ordered) list with its atoms
- `length` : number of atoms
- `adj_mat` : the adjacency matrix, corresponding to the order of the atoms list
- `truth` : either 'a' (active) or 'i' (inactive), the ground truth

Then the molecules are divided into training- and validation-set.

In a next step, the cost matrices between all possible combinations of molecules (one from the training set and one from the validation set) are calcualted.


## Instructions


## Results


## Conclusion
