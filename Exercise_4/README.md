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
- `atoms_num` : number of atoms
- `adj_mat` : the adjacency matrix, corresponding to the order of the atoms list
- `truth` : either 'a' (active) or 'i' (inactive), the ground truth
- `degree`: number of edges for each atom

Then the molecules are divided into training- and validation-set.

In a next step, the cost matrices between all possible combinations of molecules (one from the training set and one from the validation set) are calculated.

In a final step, kNN is performed for different values of k and the accuracy is printed.

## Instructions

The script `molecules.py` can be run at once and everything will be done.

In the settings part, the deletion/insertion costs for edges and nodes can be defined, as well as a list of parameters k for kNN
```python
# =============================================#
#                   Settings                   #
# =============================================#


# set the costs
Cn = 1  # cost for node deletion/insertion
Ce = 1  # cost for edge deletion/insertion

# set k-values for kNN
K = [1, 3, 5, 10, 15]
```

If the distance matrix should be saved to a file,
the part
```python
# =============================================#
#         Save the distance matrix             #
# =============================================#


# save the computed distances
np.save('dist_1_1', dist)

# note the naming: 'dist_x_y' means the distances
# have been calculated with Cn = x and Ce = y.
```
should be uncommented. Per default this part is commented, hence not active.

If the distance computations should not be done, but read from an input file, then (1) this part
```python
# =============================================#
#             Distance computation             #
# =============================================#

# compute the distance between all possible combinations of molecule pairs from valid and train (about 8 minutes)
print()
print('Computing the distance...')
start = time.time()
dist = np.array([calc_dist(mol1, mol2, Cn, Ce) for mol1, mol2 in itertools.product(valid, train)])
dist = dist.reshape((len(valid), len(train)))
end = time.time()
print('Done in {:.2f} seconds!'.format(end-start))
print()
```
should be commented (üer default this part is uncommented, hence active) and (2) this part
```python
# =============================================#
#        Read distance matrix from file        #
# =============================================#


# load the file
dist = np.load('./distances/dist_1_1.npy')

# note the naming: 'dist_x_y' means the distances
# been calculated with Cn = x and Ce = y.
```
should be uncommented (per default this part is commented, hence not active).

## Results
acc | Cn = 1 , Ce = 1
--- | ---
k = 1 | 0.992
k = 3 |0.996
k = 5 |0.996
k = 10 |0.992
k = 15 | 0.988


## Conclusion
