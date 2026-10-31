from constants import make_constants
from mesh_maker import Mesh
from solver import loop
from quadrature import quadrature

def main():
    params = make_constants(length=0.1, height=0.05, n_spines=19, layers=2)
    mesh = Mesh(params)
    quad = quadrature()
    loop(params, mesh, quad)

if __name__ == '__main__':
    main()