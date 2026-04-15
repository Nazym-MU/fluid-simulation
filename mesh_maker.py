from pprint import pprint
'''
    |~~|~~|~~|~~|~~|~~| -
    | /| /| /| /| /| /| |
    |/_|/_|/_|/_|/_|/_| |
    | /| /| /| /| /| /| m
    |/_|/_|/_|/_|/_|/_| | 
n   1  2  3  4  5  6  7 |
    |--------l--------| -
'''

class Node: # TODO: find a better name

    def __init__(self, length, m, n_spines):
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
        self.n_spines = n_spines
        self.m = m

        self.offset = self.m * 2 + 1 # TODO: come up with a better name for offset between column 1 row 1 -> column 2 row 1
        self.num_elements = self.m * (self.n_spines-1) # total number of elements (triangles)
        self.num_global_nodes = self.offset * self.n_spines

        self.l = [[0] * 6] * self.num_elements

        self.s = [[0, 0]] * self.num_global_nodes

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
            col = el // (2 * self.m) # each column pair has 2m elements
            row = el % (2 * self.m)
            start = col * 2 * self.offset + row

            self.l[el] = [start,
                          start + 2 * self.offset,
                          start + 2 * self.offset + 2,
                          start + self.offset + 1,
                          start + self.offset,
                          start + 2 * self.offset + 1]
            
            self.l[el + 1] = [start,
                              start + 2 * self.offset + 2,
                              start + 2,
                              start + 1,
                              start + self.offset + 1,
                              start + self.offset + 2]

    def coordinates_matrix(self):
        '''
        Iterate through all columns and rows to set the coordinates and populate the coordinates matrix
        '''
        for gn in range(self.num_global_nodes):
            col, row = gn // self.offset, gn % self.offset
            self.s[gn] = (col * self.length / (self.n_spines - 1), row)
        
    def element_to_global(self, element_idx):
        '''
        Returns the global node number
        '''
        return self.l[element_idx]
        
    def get_coordinates(self, global_node):
        '''
        Returns the coordinates of the global node
        '''
        return self.s[global_node] # list, so that it's mutable
    

Node(10, 2, 5)