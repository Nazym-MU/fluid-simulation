import numpy as np

def residual(mesh, quad, params, u, w, p, u_old, w_old, coords_old, delta_t):
    '''
    Writes the residual equations

    Inputs:
    - mesh: Mesh object (coords, loc_to_glob, loc_to_glob_p, ...)
    - quad: dict from quadrature()
    '''
    T = quad['T']
    W_G = quad['W_G']
    W_lG = quad['W_lG']
    phi_G = quad['phi_G']
    psi_G = quad['psi_G']
    phi1_lG = quad['phi1_lG']
    phi1_xi_lG = quad['phi1_xi_lG']
    n_Gaussian_Q = quad['n_Gaussian_Q']
    n_lGaussian_Q = quad['n_lGaussian_Q']
    Re, St, Ca = params['Re'], params['St'], params['Ca']

    # Set up Mx and Mz to calculate the sum from equation 65 and 66
    Mx, Mz = np.zeros(mesh.num_global_nodes), np.zeros(mesh.num_global_nodes)
    # Set up Continuity and KBC residuals
    C, K = np.zeros(mesh.num_p_nodes), np.zeros(mesh.num_global_nodes)

    # bulk terms first

    for ee in range(mesh.num_elements):
        nodes = mesh.loc_to_glob[ee]
        # find the x and z coordinates of the element nodes
        x_e, z_e = mesh.coords[nodes, 0], mesh.coords[nodes, 1]
        # derivatives
        x_dot_e, z_dot_e = (x_e - coords_old[nodes, 0]) / delta_t, (z_e - coords_old[nodes, 1]) / delta_t
        # velocities at element nodes
        u_e, w_e = u[nodes], w[nodes]
        # derivatives
        u_dot_e, w_dot_e = (u_e - u_old[nodes]) / delta_t, (w_e - w_old[nodes]) / delta_t


        det_Je = np.zeros(n_Gaussian_Q)
        for pp in range(n_Gaussian_Q):
            det_Je[pp] = x_e @ T[pp] @ z_e

        # x-momentum residual bulk terms
        # eq. 98
        for ii in range(6):
            for jj in range(6):
                for pp in range(n_Gaussian_Q):
                    Mx[nodes[ii]] += Re * u_dot_e[jj] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * det_Je[pp]

        # eqs. 99-100
        for ii in range(6):
            for jj in range(6):
                for kk in range(6):
                    for pp in range(n_Gaussian_Q):
                        Mx[nodes[ii]] += Re * u_e[jj] * u_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (T[pp, kk] @ z_e)
                        Mx[nodes[ii]] -= Re * x_dot_e[jj] * u_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (T[pp, kk] @ z_e)
                        Mx[nodes[ii]] += Re * w_e[jj] * u_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (-T[pp, kk] @ x_e)
                        Mx[nodes[ii]] -= Re * z_dot_e[jj] * u_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (-T[pp, kk] @ x_e)
                        

        # eq. 104: pressure term
        for ii in range(6):
            for jj in range(3):
                for pp in range(n_Gaussian_Q):
                    Mx[nodes[ii]] -= p[mesh.loc_to_glob_p[ee, jj]] * W_G[pp] * psi_G[jj, pp] * (T[pp, ii] @ z_e)

        # eq. 101-103
        for ii in range(6):
            for jj in range(6):
                for pp in range(n_Gaussian_Q):
                    Mx[nodes[ii]] += 2 * u_e[jj] * W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ z_e) / det_Je[pp]
                    Mx[nodes[ii]] += u_e[jj] * W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ x_e) / det_Je[pp]
                    Mx[nodes[ii]] -= w_e[jj] * W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ z_e) / det_Je[pp]

        
        # z-momentum residual bulk terms
        # eq. 98
        for ii in range(6):
            for jj in range(6):
                for pp in range(n_Gaussian_Q):
                    Mz[nodes[ii]] += Re * w_dot_e[jj] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * det_Je[pp]

        # eqs. 99-100
        for ii in range(6):
            for jj in range(6):
                for kk in range(6):
                    for pp in range(n_Gaussian_Q):
                        Mz[nodes[ii]] += Re * u_e[jj] * w_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (T[pp, kk] @ z_e)
                        Mz[nodes[ii]] -= Re * x_dot_e[jj] * w_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (T[pp, kk] @ z_e)
                        Mz[nodes[ii]] += Re * w_e[jj] * w_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (-T[pp, kk] @ x_e)
                        Mz[nodes[ii]] -= Re * z_dot_e[jj] * w_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (-T[pp, kk] @ x_e)

        # eq. 107: pressure term
        for ii in range(6):
            for jj in range(3):
                for pp in range(n_Gaussian_Q):
                    Mz[nodes[ii]] -= p[mesh.loc_to_glob_p[ee, jj]]  * W_G[pp] * psi_G[jj, pp] * (-T[pp, ii] @ x_e)

        # eqs. 101-102 and eq. 106
        for ii in range(6):
            for jj in range(6):
                for pp in range(n_Gaussian_Q):
                    Mz[nodes[ii]] += 2 * w_e[jj] * W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ x_e) / det_Je[pp]
                    Mz[nodes[ii]] -= u_e[jj] * W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ x_e) / det_Je[pp]
                    Mz[nodes[ii]] += w_e[jj] * W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ z_e) / det_Je[pp]

        # eq. 105: gravity term
        for ii in range(6):
            for pp in range(n_Gaussian_Q):
                Mz[nodes[ii]] += St * W_G[pp] * phi_G[ii, pp] * det_Je[pp]

        # Continuity residual bulk terms
        # eq. 67 and the approximation in eqs. 108-109
        for ii in range(3):
            for jj in range(6):
                for pp in range(n_Gaussian_Q):
                    C[mesh.loc_to_glob_p[ee, ii]] += u_e[jj] * W_G[pp] * psi_G[ii, pp] * (T[pp, jj] @ z_e)
                    C[mesh.loc_to_glob_p[ee, ii]] -= w_e[jj] * W_G[pp] * psi_G[ii, pp] * (T[pp, jj] @ x_e)

    # line integral terms next

    for ee in range(len(mesh.loc_to_glob_1)):
        nodes_1 = mesh.loc_to_glob_1[ee]
        x_le, z_le = mesh.coords[nodes_1, 0], mesh.coords[nodes_1, 1]
        u_le, w_le = u[nodes_1], w[nodes_1]
        x_dot_le, z_dot_le = (x_le - coords_old[nodes_1, 0]) / delta_t, (z_le - coords_old[nodes_1, 1]) / delta_t
        Jle = np.zeros(n_lGaussian_Q)

        # derivatives with respect to the master element parametrization:  x' and z'
        x_prime_l, z_prime_l = np.zeros(n_lGaussian_Q), np.zeros(n_lGaussian_Q)

        for pp in range(n_lGaussian_Q):
            x_prime_l[pp], z_prime_l[pp] = phi1_xi_lG[:, pp] @ x_le, phi1_xi_lG[:, pp] @ z_le

            # length of the tangent
            Jle[pp] = np.sqrt(x_prime_l[pp]**2 + z_prime_l[pp]**2)

        # x-momentum residual
        # eq. 122
        for ii in range(3):
            for pp in range(n_lGaussian_Q):
                Mx[nodes_1[ii]] += 1/Ca * W_lG[pp] * x_prime_l[pp] * phi1_xi_lG[ii, pp] / Jle[pp]

        # z-momentum residual
        # eq. 123
        for ii in range(3):
            for pp in range(n_lGaussian_Q):
                Mz[nodes_1[ii]] += 1/Ca * W_lG[pp] * z_prime_l[pp] * phi1_xi_lG[ii, pp] / Jle[pp]

        # KBC residual
        # eqs. 125-126
        for ii in range(3):
            for jj in range(3):
                for pp in range(n_lGaussian_Q):
                    K[nodes_1[ii]] -= u_le[jj] * W_lG[pp] * phi1_lG[ii, pp] * phi1_lG[jj, pp] * z_prime_l[pp]
                    K[nodes_1[ii]] += x_dot_le[jj] * W_lG[pp] * phi1_lG[ii, pp] * phi1_lG[jj, pp] * z_prime_l[pp]
                    K[nodes_1[ii]] += w_le[jj] * W_lG[pp] * phi1_lG[ii, pp] * phi1_lG[jj, pp] * x_prime_l[pp]
                    K[nodes_1[ii]] -= z_dot_le[jj] * W_lG[pp] * phi1_lG[ii, pp] * phi1_lG[jj, pp] * x_prime_l[pp]

    # return the values for the unknown indices
    return np.concatenate([Mx[mesh.u_unknown], Mz[mesh.w_unknown], C, K[mesh.surface_unknown]])