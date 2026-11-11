import numpy as np
from residual import residual

def jacobian(mesh, quad, params, u, w, p, u_old, w_old, coords_old, delta_t):
    '''
    - delta_t is the time step
    '''
    Re = params['Re']
    n_Gaussian_Q = quad['n_Gaussian_Q']
    n_lGaussian_Q = quad['n_lGaussian_Q']
    W_G = quad['W_G']
    W_lG = quad['W_lG']
    phi_G = quad['phi_G']
    psi_G = quad['psi_G']
    phi1_lG = quad['phi1_lG']
    phi1_xi_lG = quad['phi1_xi_lG']
    T = quad['T']
    n = mesh.num_global_nodes

    # derivatives of Mx

    diff_u_Mx = np.zeros((n, n))
    diff_w_Mx = np.zeros((n, n))
    diff_p_Mx = np.zeros((n, mesh.num_p_nodes))

    # derivatives of Mz

    diff_u_Mz = np.zeros((n, n))
    diff_w_Mz = np.zeros((n, n))
    diff_p_Mz = np.zeros((n, mesh.num_p_nodes))

    # derivatives of C
    diff_u_C = np.zeros((mesh.num_p_nodes, n))
    diff_w_C = np.zeros((mesh.num_p_nodes, n))

    # derivatives of K
    diff_u_K = np.zeros((n, n))
    diff_w_K = np.zeros((n, n))

    for ee in range(mesh.num_elements):
        nodes = mesh.loc_to_glob[ee]
        # x and z coordinates of the element nodes
        x_e, z_e = mesh.coords[nodes, 0], mesh.coords[nodes, 1]
        # derivatives
        x_dot_e, z_dot_e = (x_e - coords_old[nodes, 0]) / delta_t, (z_e - coords_old[nodes, 1]) / delta_t
        # velocities at element nodes
        u_e, w_e = u[nodes], w[nodes]

        det_Je = np.zeros(n_Gaussian_Q)
        for pp in range(n_Gaussian_Q):
            det_Je[pp] = x_e @ T[pp] @ z_e

        # derivative of Mx
        for ii in range(6):
            for jj in range(6):
                for kk in range(6):
                    for pp in range(n_Gaussian_Q):
                        # w.r.t u
                        diff_u_Mx[nodes[ii], nodes[jj]] += Re * u_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (T[pp, kk] @ z_e)
                        diff_u_Mx[nodes[ii], nodes[jj]] += Re * u_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[kk, pp] * (T[pp, jj] @ z_e)
                        diff_u_Mx[nodes[ii], nodes[jj]] -= Re * x_dot_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[kk, pp] * (T[pp, jj] @ z_e)
                        diff_u_Mx[nodes[ii], nodes[jj]] -= Re * w_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[kk, pp] * (T[pp, jj] @ x_e)
                        diff_u_Mx[nodes[ii], nodes[jj]] += Re * z_dot_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[kk, pp] * (T[pp, jj] @ x_e)

                        # w.r.t w
                        diff_w_Mx[nodes[ii], nodes[jj]] -= Re * u_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * T[pp, kk] @ x_e

                # terms without kk
                for pp in range(n_Gaussian_Q):
                    diff_u_Mx[nodes[ii], nodes[jj]] += Re  / delta_t * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * det_Je[pp]
                    diff_u_Mx[nodes[ii], nodes[jj]] += 2 *  W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ z_e) / det_Je[pp]
                    diff_u_Mx[nodes[ii], nodes[jj]] += W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ x_e) / det_Je[pp]
                    diff_w_Mx[nodes[ii], nodes[jj]] -= W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ z_e) / det_Je[pp]


            for jj in range(3):
                for pp in range(n_Gaussian_Q):
                    # w.r.t p
                    diff_p_Mx[nodes[ii], mesh.loc_to_glob_p[ee, jj]] -= W_G[pp] * psi_G[jj, pp] * (T[pp, ii] @ z_e)


        # derivative of Mz
        for ii in range(6):
            for jj in range(6):
                for kk in range(6):
                    for pp in range(n_Gaussian_Q):
                        # w.r.t u
                        diff_u_Mz[nodes[ii], nodes[jj]] += Re * w_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * T[pp, kk] @ z_e

                        # w.r.t w
                        diff_w_Mz[nodes[ii], nodes[jj]] += Re * u_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[kk, pp] * (T[pp, jj] @ z_e)
                        diff_w_Mz[nodes[ii], nodes[jj]] -= Re * x_dot_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[kk, pp] * (T[pp, jj] @ z_e)
                        diff_w_Mz[nodes[ii], nodes[jj]] -= Re * w_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (T[pp, kk] @ x_e)
                        diff_w_Mz[nodes[ii], nodes[jj]] -= Re * w_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[kk, pp] * (T[pp, jj] @ x_e)
                        diff_w_Mz[nodes[ii], nodes[jj]] += Re * z_dot_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[kk, pp] * (T[pp, jj] @ x_e)

                # terms without kk
                for pp in range(n_Gaussian_Q):
                    diff_u_Mz[nodes[ii], nodes[jj]] -= W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ x_e) / det_Je[pp]
                    diff_w_Mz[nodes[ii], nodes[jj]] += Re  / delta_t * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * det_Je[pp]
                    diff_w_Mz[nodes[ii], nodes[jj]] += 2 *  W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ x_e) / det_Je[pp]
                    diff_w_Mz[nodes[ii], nodes[jj]] += W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ z_e) / det_Je[pp]

            for jj in range(3):
                for pp in range(n_Gaussian_Q):
                    # w.r.t p
                    diff_p_Mz[nodes[ii], mesh.loc_to_glob_p[ee, jj]] += W_G[pp] * psi_G[jj, pp] * (T[pp, ii] @ x_e)


        # derivative of C
        for ii in range(3):
            for jj in range(6):
                for pp in range(n_Gaussian_Q):
                    # w.r.t u
                    diff_u_C[mesh.loc_to_glob_p[ee, ii], nodes[jj]] += W_G[pp] * psi_G[ii, pp] * T[pp, jj] @ z_e
                    # w.r.t w
                    diff_w_C[mesh.loc_to_glob_p[ee, ii], nodes[jj]] -= W_G[pp] * psi_G[ii, pp] * T[pp, jj] @ x_e


    for ee in range(len(mesh.loc_to_glob_1)):
        nodes_1 = mesh.loc_to_glob_1[ee]
        x_le, z_le = mesh.coords[nodes_1, 0], mesh.coords[nodes_1, 1]
        Jle = np.zeros(n_lGaussian_Q)


        # derivatives with respect to the master element parametrization:  x' and z'
        x_prime_l, z_prime_l = np.zeros(n_lGaussian_Q), np.zeros(n_lGaussian_Q)
        for pp in range(n_lGaussian_Q):
            x_prime_l[pp], z_prime_l[pp] = phi1_xi_lG[:, pp] @ x_le, phi1_xi_lG[:, pp] @ z_le

            # length of the tangent
            Jle[pp] = np.sqrt(x_prime_l[pp]**2 + z_prime_l[pp]**2)
        
        
        # derivative of K
        for ii in range(3):
            for jj in range(3):
                for pp in range(n_lGaussian_Q):
                    # w.r.t u
                    diff_u_K[nodes_1[ii], nodes_1[jj]] -= W_lG[pp] * phi1_lG[ii, pp] * phi1_lG[jj, pp] * z_prime_l[pp]
                    # w.r.t w
                    diff_w_K[nodes_1[ii], nodes_1[jj]] += W_lG[pp] * phi1_lG[ii, pp] * phi1_lG[jj, pp] * x_prime_l[pp]


    # creating the Jacobian
    JR = np.zeros()