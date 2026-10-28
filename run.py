from constants import make_constants
from mesh_maker import Mesh
from solver import loop

def main():
    params = make_constants(length=0.1, height=0.05, n_spines=19, layers=2)
    mesh = Mesh(params)
    loop(params, mesh)

if __name__ == '__main__':
    main()