# Exercise 4 - Molecules


## Packages
- xml.etree.ElementTree
- numpy 1.12.0
- scipy 0.19.0
- itertools
- time


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
this snippet represents the (nonsense) molecule with two atoms, P and O, bonded with valence 1. So a molecule is represented as a graph: atoms are edges, where their labels are their chemical symbol, and the covalent bonds between atoms are edges, where their labels are the valence of their linkage (single or double). For this task, we don't distinguish between single or double linkages between atoms. We only consider if two atoms are connected or not. Hence we ignore the edge label.


## Description

In this exercise, the molecules of the validation set are classified using KNN with the approximate graph edit distance (aGED). The validation set and the training set contain 250 molecules each.

In a first step, the provided dataset is converted into an appropriate data structure. Each molecule is represented as an object of class `Molecule` with the following attributes
- `id` : the molecules identification number
- `atoms` : a (ordered) list with the molecules atoms
- `atoms_num` : number of total atoms
- `adj_mat` : the adjacency matrix, corresponding to the order of elements in `atoms`, either 0 for not linked or 1 for linked
- `truth` : the ground truth, either 1 for active, 0 for inactive or -1 for undefined
- `degree`: number of edges for each atom (corresponding to the order of elements in `atoms`)

And the molecules are divided into training- and validation-set.

In a second step, the aGED between all possible combinations of molecules (one from the training set and one from the validation set) are calculated by solving the [assignment problem](https://en.wikipedia.org/wiki/Assignment_problem) with the cost matrix C, using the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) from the  [scipy.optimize framework](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html). For further elaborations on the calculations of the entries of the matrix see, see section [Calculating cost matrix C](#calculating-cost-matrix-c).

In a third and final step, kNN is performed for different values of k and the accuracy is printed. As distance to compare two molecules, the (approximate) graph edit distance is used.


## Calculating cost matrix C

Let `n` be the number of atoms of molecule 1 and `m` the number of atoms of molecule 2.

The `(n+m)x(n+m)` cost matrix `C` has four parts:
1. Upper left: substitutions (nodes plus adjacent edges)  ->  `c_i,j` for `i` in 1..n and `j` in 1..m
2. Upper right: deletions (nodes plus adjacent edges)  ->  `c_i,eps` for `i` in 1..n on the diagonal, rest is 99999
3. Lower left: insertions (nodes plus adjacent edges)  ->  `c_eps,j` for `j` in 1..m on the diagonal, rest is 99999
4. Lower right: dummy assignments (eps -> eps)  ->  0

Assume `u_i` to be a node from molecule 1 and `v_i` a node from molecule 2.
Let `P_i` be the set of all adjacent edges to `u_i` and `Q_i` the set of all adjacent edges to `v_i`.

Then, the entries (`c_i,j` / `c_i,eps` / `c_eps,j`) of the cost matrix `C` are:
- Deletion costs include the deletion of the node `u_i` as well as the deletion of all edges in `P_i` :

      c_i,eps = c(u_i -> eps) + sum_{p in P_i} c(p -> eps)

- Insertion costs include the insertion of the node `v_i` as well as the insertion of all edges in `Q_i` :

      c_eps,j = c(eps -> v_j) + sum_{q in Q_i} c(eps -> q)

- Substitution costs include the node substitution `c(u_i -> v_i)` as well as an estimation of the
  edge assignment cost `C(P_i -> Q_j)` :

      c_i,j = c(u_i -> v_j) + C(P_i -> Q_j)

For insertion and deletion cost we consider the Dirac cost function, where we fix two positive numbers `Ce` and `Cn`:
- node substitution: `c(u_i -> v_j) = 2*Cn` if symbols are not equal, `c(u_i -> v_j) = 0` otherwise
- node deletion/insertion: `c(u_i -> eps) = c(eps -> v_j) = Cn`
- edge deletion/insertion: `c(p -> eps) = c(eps -> q) = Ce`

The estimation of the edge assignment costs `C(P_i -> Q_j)` are assumed to be the number of edges that have to be
either deleted or inserted to get from `P_i` to `Q_j`. Hence `C(P_i -> Q_j)` is the absolute value of the difference in node number between the two sets: `abs(|P_i|-|Q_j|)`.

Combining this all together:
- `c_i,eps = Cn + |P_i|*Ce`
- `c_eps,j = Cn + |Q_j]*Ce`
- `c_i,j = 2*Cn + abs(|P_i|-|Q_j|)` if the symbols are equal and `c_i,j = abs(|P_i|-|Q_j|)` otherwise


## Instructions

The script `molecules.py` can be run at once and everything will be done. It takes about 8 minutes.

In the settings part, the deletion/insertion costs for edges and nodes can be defined, as well as a list of parameters k for kNN
```python
# =============================================#
#                   Settings                   #
# =============================================#


# set the costs
Cn = 1  # cost for node deletion/insertion
Ce = 3  # cost for edge deletion/insertion

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
np.save('dist_1_3', dist)

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
dist = np.load('./distances/dist_1_3.npy')

# note the naming: 'dist_x_y' means the distances
# been calculated with Cn = x and Ce = y.
```
should be uncommented (per default this part is commented, hence not active).


## Results

The accuracy is displayed in the following table

k | Cn=1 Ce=1 | Cn=1 Ce=2 | Cn=1 Ce=3 | Cn=1 Ce=4 | Cn=2 Ce=1 | Cn=3 Ce=1 | Cn=4 Ce=1
--- | --- | --- | --- | --- | --- | --- | ---
**1** | 0.992 | 0.992| 0.996 | 0.996 | 0.988 | 0.988 | 0.988
**3** |0.996 | 0.996 | 0.996 | 0.996 | 0.992 | 0.992 | 0.992
**5** |0.996 | 0.996 | 0.996 | 0.996 | 0.992 | 0.992 | 0.992
**10** |0.992 | 0.992 | 0.992 | 0.992 | 0.988 | 0.988 | 0.988
**15** | 0.988 | 0.992 | 0.992 | 0.992 | 0.988 | 0.988 | 0.988


## Conclusion

For the tested combinations of `k`, `Cn` and `Ce`, we get accuracies ranging from 0.988 to 0.996. So the accuracy is over all pretty good.

For all choices of `Cn` and `Ce`, `k=3` and `k=5` yield the best accuracy.
For `Cn=1` & `Ce=3` and `Cn=1` & `Ce=4`, `k=1` yields the same accuracy as `k=3` and `k=5`.

For `k=3` and `k=5`, the combinations `Cn=1` & `Ce=1`, `Cn=1` & `Ce=2`, `Cn=1` & `Ce=3` and `Cn=1` & `Ce=4` all yield the same best accuracy of 0.996.

So, edge insertion/deletions should cost either the same as node insertion/deletions, or a multiple.
