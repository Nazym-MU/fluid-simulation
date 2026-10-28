import numpy as np
import matplotlib.pyplot as plt
'''
    |~~|~~|~~|~~|~~|~~| -
    | /| /| /| /| /| /| |
    |/_|/_|/_|/_|/_|/_| |
    | /| /| /| /| /| /| m
    |/_|/_|/_|/_|/_|/_| | 
n   1  2  3  4  5  6  7 |
    |--------L--------| -
'''

class Mesh:

    def __init__(self, params):
        '''
        TODO: write the docstring
        '''
        self.length = params['L']
        self.height = params['H']
        self.n_spines = params['n_spines']
        self.layers = params['layers']

        self.nodes_per_col = self.layers * 2 + 1
        self.num_elements = self.layers * (self.n_spines-1) # total number of elements (triangles)
        self.num_global_nodes = self.nodes_per_col * self.n_spines
        self.num_p_nodes = (self.n_spines + 1) // 2 * (self.layers + 1)

        self.loc_to_glob = np.zeros((self.num_elements, 6), dtype=int)
        self.loc_to_glob_p = np.zeros((self.num_elements, 3), dtype=int) # lp: map from local node number to pressure-node number
        self.coords = np.zeros((self.num_global_nodes, 2))

        self.element_matrix()
        self.unknown_nodes()
        self.coordinates_matrix()

    
    def element_matrix(self):
        r'''
            2
            |\ 
           3| \ 5
            |__\ 
            0 4 1
        '''
        n = self.nodes_per_col
        cols = np.arange((self.n_spines - 1) // 2)
        rows = np.arange(self.layers)

        # velocity node numbering
        start = (2*n*cols[:, None] + 2*rows[None, :]).flatten()
        lower = np.array([0, 2*n, 2*n+2, n+1, n, 2*n+1])
        upper = np.array([0, 2*n+2, 2, 1, n+1, n+2])
        self.loc_to_glob[0::2] = start[:, None] + lower
        self.loc_to_glob[1::2] = start[:, None] + upper

        # pressure node numbering
        m = self.layers + 1
        start_p = (m*cols[:, None] + rows[None, :]).flatten()
        lower_p = np.array([0, m, m+1])
        upper_p = np.array([0, m+1, 1])
        self.loc_to_glob_p[0::2] = start_p[:, None] + lower_p
        self.loc_to_glob_p[1::2] = start_p[:, None] + upper_p

        # free surface node numbering
        self.surface_nodes = np.arange(1, self.n_spines + 1) * n - 1   # all nodes on ∂Ω¹, left to right
        self.loc_to_glob_1 = np.column_stack([self.surface_nodes[:-1:2],
                                        self.surface_nodes[1::2],
                                        self.surface_nodes[2::2]])
            
    def surface_elevation(self, x):
        '''
        A sine wave function returns its elevation
        '''
        return 0.01 * np.sin(2 * np.pi * x / self.length)

    def coordinates_matrix(self):
        '''
        Set the coordinates and populate the coordinates matrix
        '''
        gn = np.arange(self.num_global_nodes)
        col, row = gn // self.nodes_per_col, gn % self.nodes_per_col
        x = col * self.length / (self.n_spines - 1)
        zeta = self.surface_elevation(x)
        z = -self.height + row / (2*self.layers) * (self.height + zeta)
        self.coords = np.column_stack([x, z])

    def unknown_nodes(self):
        '''
        Creates a list of indices of unknown velocities,
        i.e. every node except for those at the walls and at the bottom (no-slip, u = 0, w = 0 at those points)
        '''
        gn = np.arange(self.num_global_nodes) # global nodes
        n = self.nodes_per_col

        col, row = gn // n, gn % n
        on_wall = (col == 0) | (col == self.n_spines - 1) | (row == 0)
        self.u_unknown = self.w_unknown = gn[~on_wall]
        self.p_unknown = np.arange(self.num_p_nodes)
        self.surface_unknown = self.surface_nodes[1:-1]

    def plot(self, labels=True):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_aspect('equal')

        xs, zs = self.coords[:, 0], self.coords[:, 1]

        # triangle edges (vertices only, straight lines)
        ax.triplot(xs, zs, self.loc_to_glob[:, :3], color='b', lw=1)

        # all velocity nodes
        ax.plot(xs, zs, 'ko', ms=4)

        if labels:
            for gn in range(self.num_global_nodes):
                ax.text(xs[gn], zs[gn], str(gn), fontsize=8,
                        ha='center', va='bottom', color='purple')

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title('Mesh')
        plt.show()


    
if __name__ == '__main__':
    from constants import make_constants
    Mesh(make_constants(0.1, 0.05, 19, 2)).plot()