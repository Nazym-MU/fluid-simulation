from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
'''
    |~~|~~|~~|~~|~~|~~| -
    | /| /| /| /| /| /| |
    |/_|/_|/_|/_|/_|/_| |
    | /| /| /| /| /| /| m
    |/_|/_|/_|/_|/_|/_| | 
n   1  2  3  4  5  6  7 |
    |--------l--------| -
'''

class Mesh:

    def __init__(self, length, height, layers, n_spines):
        '''
        Creates 2 matrices for mapping elements:
        - l: matrix mapping (element_idx, local velocity node number) to its global node number
        - s: matrix referencing the coordinates of the global element
        ------------------------------------
        Inputs:
        length: the length of the container
        m: the number of triangles in a single column
        n_spines: the number of spine (must be an odd number)

        Returns:
        2 matrices l and c
        '''
        self.length = length
        self.height = height
        self.n_spines = n_spines
        self.layers = layers

        self.nodes_per_col = self.layers * 2 + 1
        self.num_elements = self.layers * (self.n_spines-1) # total number of elements (triangles)
        self.num_global_nodes = self.nodes_per_col * self.n_spines

        self.loc2glob = [[0] * 6] * self.num_elements

        self.coords = [[0, 0]] * self.num_global_nodes

        self.element_matrix()
        self.coordinates_matrix()

        pprint(self.l)

    
    def element_matrix(self):
        '''
            2
            |\ 
           3| \ 5
            |__\ 
            0 4 1
        '''
        # TODO: question – do we always have even number of triangles in two columns? edge cases
        for el in range(0, self.num_elements, 2):
            col = el // (2 * self.layers) # each column pair has 2m elements
            row = el % (2 * self.layers)
            start = col * 2 * self.nodes_per_col + row

            self.loc2globl[el] = [start,
                          start + 2 * self.nodes_per_col,
                          start + 2 * self.nodes_per_col + 2,
                          start + self.nodes_per_col + 1,
                          start + self.nodes_per_col,
                          start + 2 * self.nodes_per_col + 1]
            
            self.loc2glob[el + 1] = [start,
                              start + 2 * self.nodes_per_col + 2,
                              start + 2,
                              start + 1,
                              start + self.nodes_per_col + 1,
                              start + self.nodes_per_col + 2]
            
    def spine_height(self, i):
        '''
        A sine wave function that takes in ith spine (horizontally) and returns its height
        '''
        if i < self.n_spines:
            x_coord = i * self.length / (self.n_spines - 1)
            return self.height + np.sin(x_coord) # height of the spine varies around the initial height

    def coordinates_matrix(self):
        '''
        Iterate through all columns and rows to set the coordinates and populate the coordinates matrix
        '''
        for gn in range(self.num_global_nodes):
            col, row = gn // self.nodes_per_col, gn % self.nodes_per_col
            h = self.spine_height(col)
            self.coords[gn] = (col * self.length / (self.n_spines - 1), row / (2 * self.layers) * h)
        
    def element_to_global(self, element_idx):
        '''
        Returns the global node number
        '''
        return self.loc2glob[element_idx]
        
    def get_coordinates(self, global_node):
        '''
        Returns the coordinates of the global node
        '''
        return self.coords[global_node] # list, so that it's mutable
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_aspect('equal')

        xs = [coord[0] for coord in self.coords]
        ys = [coord[1] for coord in self.coords]

        for nodes in self.loc2glob:
            for a, b in [(0,1),(1,2),(2,0)]:
                ax.plot([xs[nodes[a]], xs[nodes[b]]], [ys[nodes[a]], ys[nodes[b]]], 'b-', lw=1)
            for gn in nodes:
                ax.plot(xs[gn], ys[gn], 'ko', ms=4)
                ax.text(xs[gn], ys[gn], str(gn), fontsize=9, ha='center', va='bottom', color='purple')

        ax.set_title('Mesh')
        plt.show()


    

Mesh(10, 6, 2, 19).plot()