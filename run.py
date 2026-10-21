from constants import make_constants
from mesh_maker import Mesh
from solver import loop

def main():
    params = make_constants()
    mesh = Mesh
    loop(params, mesh)

if __name__ == '__main__':
    main()