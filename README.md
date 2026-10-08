# fluid-simulation

A finite element solver for free-surface waves in a 2D pool, based on an Arbitrary Lagrangian–Eulerian (ALE) formulation. The fluid sits in a rectangular tank with no-slip walls and bottom; the top is a free surface pinned at the rims. Because the surface moves, the mesh has to move with it, so the domain is part of the solution. This is my senior capstone at Minerva University, advised by Prof. Carlos Galeano-Ríos.

So far the repo contains the mesh generator, `mesh_maker.py` (and the same code in `mesh_maker.ipynb`). Everything else in the solver (residuals, Jacobian, Newton loop) is built on top of it.

## How the mesh works

**Spines.** The domain is cut by `n_spines` vertical lines. Every node sits on one of them, so when the free surface rises or falls the nodes slide up and down their spine and each column of the mesh simply stretches or shrinks. `n_spines` must be odd: every other spine carries the edge midpoints of the triangles.

**Nodes.** Each spine holds `2 * layers + 1` nodes, evenly spaced from the bottom to the surface. Global node numbers go bottom to top within a spine, then move one spine to the right, starting from 0. The extra nodes between vertices are there because the velocity is quadratic (Taylor–Hood P2–P1): each triangle needs 6 velocity nodes, at the three vertices and the three edge midpoints. Pressure is linear and only uses the vertices.

**Elements.** Each rectangular cell between two vertex spines is split along its diagonal into two triangles. The triangle containing the bottom-right corner is numbered first. Within a triangle the local nodes are ordered the same way as on the master element: 0, 1, 2 are the vertices counter-clockwise, and 3, 4, 5 are the midpoints of edges 0–2, 0–1 and 1–2. Keeping this order identical everywhere is what lets one set of shape functions work for every element.

**`loc2glob`.** `loc2glob[e][jj]` is the global node number of local node `jj` in element `e` (the map $l(e, jj)$ in the write-up). The residuals are computed element by element, and this table says which global unknowns each local contribution belongs to.

**`coords`.** `coords[n]` is the `(x, z)` position of global node `n`. These are the node positions $\mathcal X_j, \mathcal Z_j$ that feed the map from the master triangle to each physical element, and from it the element Jacobian.

**`spine_height`.** Sets the initial height of each spine, currently a sine wave around `height`. This is a placeholder for the initial surface disturbance; in the full solver the surface heights become unknowns.

**`plot`.** Draws the triangles with their global node numbers, to check the numbering by eye.

## Running it

```bash
pip install numpy matplotlib
python mesh_maker.py
```

`Mesh(length, height, layers, n_spines)` builds the mesh; the script ends with `Mesh(10, 6, 2, 19).plot()`.
