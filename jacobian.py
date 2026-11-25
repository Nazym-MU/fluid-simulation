import numpy as np
from residual import residual

def jacobian(mesh, quad, params, u, w, p, u_old, w_old, coords_old, delta_t):
    '''
    - delta_t is the time step
    '''
    Re = params['Re']
    Ca = params['Ca']
    St = params['St']
    n_Gaussian_Q = quad['n_Gaussian_Q']
    n_lGaussian_Q = quad['n_lGaussian_Q']
    W_G = quad['W_G']
    W_lG = quad['W_lG']
    phi_G = quad['phi_G']
    psi_G = quad['psi_G']
    phi1_lG = quad['phi1_lG']
    diff_xi_phi1_lG = quad['diff_xi_phi1_lG']
    T = quad['T']
    n = mesh.num_global_nodes

    # derivatives of Mx

    diff_u_Mx = np.zeros((n, n))
    diff_w_Mx = np.zeros((n, n))
    diff_p_Mx = np.zeros((n, mesh.num_p_nodes))
    diff_h_Mx = np.zeros((n, mesh.n_spines - 2))

    # derivatives of Mz

    diff_u_Mz = np.zeros((n, n))
    diff_w_Mz = np.zeros((n, n))
    diff_p_Mz = np.zeros((n, mesh.num_p_nodes))
    diff_h_Mz = np.zeros((n, mesh.n_spines - 2))

    # derivatives of C
    diff_u_C = np.zeros((mesh.num_p_nodes, n))
    diff_w_C = np.zeros((mesh.num_p_nodes, n))
    diff_h_C = np.zeros((mesh.num_p_nodes, mesh.n_spines - 2))

    # derivatives of K
    diff_u_K = np.zeros((n, n))
    diff_w_K = np.zeros((n, n))
    diff_h_K = np.zeros((n, mesh.n_spines - 2))

    row = np.arange(n) % mesh.nodes_per_col
    spine = np.arange(n) // mesh.nodes_per_col
    diff_h_z = row / (mesh.nodes_per_col - 1) # eq. 145

    for ee in range(mesh.num_elements):
        nodes = mesh.loc_to_glob[ee]
        # x and z coordinates of the element nodes
        x_e, z_e = mesh.coords[nodes, 0], mesh.coords[nodes, 1]
        # time derivatives
        x_dot_e, z_dot_e = (x_e - coords_old[nodes, 0]) / delta_t, (z_e - coords_old[nodes, 1]) / delta_t
        # partial derivative of z_e w.r.t h
        diff_h_z_e = diff_h_z[nodes]

        # velocities at element nodes
        u_e, w_e = u[nodes], w[nodes]
        # derivatives of velocities
        u_dot_e, w_dot_e = (u[nodes] - u_old[nodes]) / delta_t, (w[nodes] - w_old[nodes]) / delta_t

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
                    # w.r.t u
                    diff_u_Mx[nodes[ii], nodes[jj]] += Re  / delta_t * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * det_Je[pp]
                    diff_u_Mx[nodes[ii], nodes[jj]] += 2 *  W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ z_e) / det_Je[pp]
                    diff_u_Mx[nodes[ii], nodes[jj]] += W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ x_e) / det_Je[pp]
                    # w.r.t w
                    diff_w_Mx[nodes[ii], nodes[jj]] -= W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ z_e) / det_Je[pp]
                    
            for jj in range(3):
                for pp in range(n_Gaussian_Q):
                    # w.r.t p
                    diff_p_Mx[nodes[ii], mesh.loc_to_glob_p[ee, jj]] -= W_G[pp] * psi_G[jj, pp] * (T[pp, ii] @ z_e)

        # w.r.t h
        '''
        %Find the derivative of each of the nodal positions with 
        % respect to each of the spines that affect this particular 
        %element
        diff_h_r_e = zeros(6,n_s_e);
        diff_h_z_e = zeros(6,n_s_e);
        diff_h_r_e(:,1) = diff_h0_rz(l(ee,:),1);
        diff_h_z_e(:,1) = diff_h0_rz(l(ee,:),2);
        for qq = 2:n_s_e
            for rr = el_s_loc_nodes{ee,qq}
                diff_h_r_e(rr,qq) = diff_h_rz(l(ee,rr),1);
                diff_h_z_e(rr,qq) = diff_h_rz(l(ee,rr),2);
            end
        end
        '''
        spine_e = spine[nodes]
        for qq in range(1, mesh.n_spines - 1):  # skip the first and the last spines because they're fixed
            diff_hq_z_e = diff_h_z_e * (spine_e == qq)
            diff_det_Je = np.array([x_e @ T[pp] @ diff_hq_z_e for pp in range(n_Gaussian_Q)])

            for ii in range(6):
                for jj in range(6):
                    for kk in range(6):
                        for pp in range(n_Gaussian_Q):
                            diff_h_Mx[nodes[ii], qq - 1] += Re * u_e[jj] * u_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[kk, pp] * (T[pp, jj] @ diff_hq_z_e)
                            diff_h_Mx[nodes[ii], qq - 1] += Re / delta_t * diff_hq_z_e[jj] * u_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (T[pp, kk] @ x_e)

                    # terms without kk
                    for pp in range(n_Gaussian_Q):
                        diff_h_Mx[nodes[ii], qq - 1] += Re * u_dot_e[jj] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * diff_det_Je[pp]
                        diff_h_Mx[nodes[ii], qq - 1] += 2 * u_e[jj] * W_G[pp] * (T[pp, ii] @ diff_hq_z_e) * (T[pp, jj] @ z_e) / det_Je[pp]
                        diff_h_Mx[nodes[ii], qq - 1] += 2 * u_e[jj] * W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ diff_hq_z_e) / det_Je[pp]
                        diff_h_Mx[nodes[ii], qq - 1] -= 2 * u_e[jj] * W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ z_e) * diff_det_Je[pp] / (det_Je[pp])**2
                        diff_h_Mx[nodes[ii], qq - 1] -= u_e[jj] * W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ x_e) * diff_det_Je[pp] / det_Je[pp]**2
                        diff_h_Mx[nodes[ii], qq - 1] -= w_e[jj] * W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ diff_hq_z_e) / det_Je[pp]
                        diff_h_Mx[nodes[ii], qq - 1] += w_e[jj] * W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ z_e) * diff_det_Je[pp] / (det_Je[pp])**2

                for jj in range(3):
                    for pp in range(n_Gaussian_Q):
                        diff_h_Mx[nodes[ii], qq - 1] -= p[mesh.loc_to_glob_p[ee, jj]] * W_G[pp] * psi_G[jj, pp] * T[pp, ii] @ diff_hq_z_e


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

        # w.r.t h
        for qq in range(1, mesh.n_spines - 1):
            diff_hq_z_e = diff_h_z_e * (spine_e == qq)
            diff_det_Je = np.array([x_e @ T[pp] @ diff_hq_z_e for pp in range(n_Gaussian_Q)])

            for ii in range(6):
                for jj in range(6):
                    for kk in range(6):
                        for pp in range(n_Gaussian_Q):
                            diff_h_Mz[nodes[ii], qq - 1] += Re * u_e[jj] * w_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (T[pp, kk] @ diff_hq_z_e)
                            diff_h_Mz[nodes[ii], qq - 1] += Re / delta_t * diff_hq_z_e[jj] * w_e[kk] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * (T[pp, kk] @ x_e)

                    # terms without kk
                    for pp in range(n_Gaussian_Q):
                        diff_h_Mz[nodes[ii], qq - 1] += Re * w_dot_e[jj] * W_G[pp] * phi_G[ii, pp] * phi_G[jj, pp] * diff_det_Je[pp]
                        diff_h_Mz[nodes[ii], qq - 1] -= 2 * w_e[jj] * W_G[pp] * (T[pp, ii] @ x_e) * (T[pp, jj] @ x_e) * diff_det_Je[pp] / (det_Je[pp])**2
                        diff_h_Mz[nodes[ii], qq - 1] -= u_e[jj] * W_G[pp] * (T[pp, ii] @ diff_hq_z_e) * (T[pp, jj] @ x_e) / det_Je[pp]
                        diff_h_Mz[nodes[ii], qq - 1] += u_e[jj] * W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ x_e) * diff_det_Je[pp] / (det_Je[pp])**2
                        diff_h_Mz[nodes[ii], qq - 1] += w_e[jj] * W_G[pp] * (T[pp, ii] @ diff_hq_z_e) * (T[pp, jj] @ z_e) / det_Je[pp]
                        diff_h_Mz[nodes[ii], qq - 1] += w_e[jj] * W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ diff_hq_z_e) / det_Je[pp]
                        diff_h_Mz[nodes[ii], qq - 1] -= w_e[jj] * W_G[pp] * (T[pp, ii] @ z_e) * (T[pp, jj] @ z_e) * diff_det_Je[pp] / (det_Je[pp])**2

                # gravity
                for pp in range(n_Gaussian_Q):
                    diff_h_Mz[nodes[ii], qq - 1] += St * W_G[pp] * phi_G[ii, pp] * diff_det_Je[pp]


        # derivative of C
        for ii in range(3):
            for jj in range(6):
                for pp in range(n_Gaussian_Q):
                    # w.r.t u
                    diff_u_C[mesh.loc_to_glob_p[ee, ii], nodes[jj]] += W_G[pp] * psi_G[ii, pp] * T[pp, jj] @ z_e
                    # w.r.t w
                    diff_w_C[mesh.loc_to_glob_p[ee, ii], nodes[jj]] -= W_G[pp] * psi_G[ii, pp] * T[pp, jj] @ x_e

        # w.r.t h
        for qq in range(1, mesh.n_spines - 1):
            diff_hq_z_e = diff_h_z_e * (spine_e == qq)

            for ii in range(3):
                for jj in range(6):
                    for pp in range(n_Gaussian_Q):
                        diff_h_C[mesh.loc_to_glob_p[ee, ii], qq - 1] += u_e[jj] * W_G[pp] * psi_G[ii, pp] * T[pp, jj] @ diff_hq_z_e

    for ee in range(len(mesh.loc_to_glob_1)):
        nodes_1 = mesh.loc_to_glob_1[ee]
        x_le, z_le = mesh.coords[nodes_1, 0], mesh.coords[nodes_1, 1]
        u_le = u[nodes_1]
        J_le_1 = np.zeros(n_lGaussian_Q)

        # derivatives with respect to the master element parametrization:  x' and z'
        diff_xi_x_l, diff_xi_z_l = np.zeros(n_lGaussian_Q), np.zeros(n_lGaussian_Q)
        for pp in range(n_lGaussian_Q):
            diff_xi_x_l[pp], diff_xi_z_l[pp] = diff_xi_phi1_lG[:, pp] @ x_le, diff_xi_phi1_lG[:, pp] @ z_le

            # length of the tangent
            J_le_1[pp] = np.sqrt(diff_xi_x_l[pp]**2 + diff_xi_z_l[pp]**2)
        
        # derivative of K
        for ii in range(3):
            for jj in range(3):
                for pp in range(n_lGaussian_Q):
                    # w.r.t u
                    diff_u_K[nodes_1[ii], nodes_1[jj]] -= W_lG[pp] * phi1_lG[ii, pp] * phi1_lG[jj, pp] * diff_xi_z_l[pp]
                    # w.r.t w
                    diff_w_K[nodes_1[ii], nodes_1[jj]] += W_lG[pp] * phi1_lG[ii, pp] * phi1_lG[jj, pp] * diff_xi_x_l[pp]

        # line terms:
        spine_l = spine[nodes_1]
        for qq in range(1, mesh.n_spines - 1):
            diff_hq_z_l = diff_h_z[nodes_1] * (spine_l == qq)
            # derivative of diff_xi_z_l: matmul of shapes (1, 3) vector @ (3, 11) matrix = (1, 11) vector
            # derivative w.r.t hq at each quadrature point
            diff_hq_diff_xi_z_l = diff_hq_z_l @ diff_xi_phi1_lG
            # derivative of J_l^1 w.r.t hq
            diff_hq_J_le_1 = diff_xi_z_l * diff_hq_diff_xi_z_l / J_le_1

            for pp in range(n_lGaussian_Q):
                for ii in range(3):
                    # x momentum
                    diff_h_Mx[nodes_1[ii], qq - 1] -= 1/Ca * W_lG[pp] * diff_xi_x_l[pp] * diff_hq_J_le_1[pp] / J_le_1[pp]**2 * diff_xi_phi1_lG[ii, pp]
                    # z momentum
                    diff_h_Mz[nodes_1[ii], qq - 1] += 1/Ca * W_lG[pp] * diff_hq_diff_xi_z_l[pp] / J_le_1[pp] * diff_xi_phi1_lG[ii, pp]
                    diff_h_Mz[nodes_1[ii], qq - 1] -= 1/Ca * W_lG[pp] * diff_xi_z_l[pp] * diff_hq_J_le_1[pp] / J_le_1[pp]**2 * diff_xi_phi1_lG[ii, pp]
                    # KBC
                    for jj in range(3):
                        diff_h_K[nodes_1[ii], qq - 1] -= u_le[jj] * W_lG[pp] * phi1_lG[ii, pp] * phi1_lG[jj, pp] * diff_hq_diff_xi_z_l[pp]
                        diff_h_K[nodes_1[ii], qq - 1] -= 1/delta_t * (diff_hq_z_l[jj]) * W_lG[pp] * phi1_lG[ii, pp] * phi1_lG[jj, pp] * diff_xi_x_l[pp]



    # creating the Jacobian (a square matrix)

    # number of rows:
    # = unknown Mx [all nodes - wall nodes], i.e. u_unknown
    # + unknown Mz [all nodes - wall nodes], i.e. w_unknown
    # + unknown C [all pressure nodes], i.e. p_unknown
    # + unknown K [all surface nodes - wall nodes], i.e. surface_unknown
    
    # number of cols: u_unknown + w_unknown + p_unknown + surface_unknown
    
    n_u_uk = len(mesh.u_unknown)
    n_w_uk = len(mesh.w_unknown)
    n_p_uk = len(mesh.p_unknown)
    n_surface_uk = len(mesh.surface_unknown)

    row_u, row_w, row_p, row_h = 0, n_u_uk, n_u_uk + n_w_uk, n_u_uk + n_w_uk + n_p_uk
    col_u, col_w, col_p, col_h = 0, n_u_uk, n_u_uk + n_w_uk, n_u_uk + n_w_uk + n_p_uk

    JR = np.zeros((n_u_uk + n_w_uk + n_p_uk + n_surface_uk,
                  n_u_uk + n_w_uk + n_p_uk + n_surface_uk))

# 1st block row of the Jacobian (Mx)
    # derivative of Mx w.r.t u
    JR[row_u:row_w, col_u:col_w] = diff_u_Mx[mesh.u_unknown][:, mesh.u_unknown]
    # derivative of Mx w.r.t w
    JR[row_u:row_w, col_w:col_p] = diff_w_Mx[mesh.u_unknown][:, mesh.w_unknown]
    # derivative of Mx w.r.t p
    JR[row_u:row_w, col_p:col_h] = diff_p_Mx[mesh.u_unknown][:, mesh.p_unknown]
    # derivative of Mx w.r.t h
    JR[row_u:row_w, col_h:] =  diff_h_Mx[mesh.u_unknown]

# 2nd block row of the Jacobian (Mz)
    # derivative of Mz w.r.t u
    JR[row_w:row_p, col_u:col_w] = diff_u_Mz[mesh.w_unknown][:, mesh.u_unknown]
    # derivative of Mz w.r.t w
    JR[row_w:row_p, col_w:col_p] = diff_w_Mz[mesh.w_unknown][:, mesh.w_unknown]
    # derivative of Mz w.r.t p
    JR[row_w:row_p, col_p:col_h] = diff_p_Mz[mesh.w_unknown][:, mesh.p_unknown]
    # derivative of Mz w.r.t h
    JR[row_w:row_p, col_h:] =  diff_h_Mz[mesh.w_unknown]

# 3rd block row of the Jacobian (C)
    # derivative of C w.r.t u
    JR[row_p:row_h, col_u:col_w] = diff_u_C[mesh.p_unknown][:, mesh.u_unknown]
    # derivative of C w.r.t w
    JR[row_p:row_h, col_w:col_p] = diff_w_C[mesh.p_unknown][:, mesh.w_unknown]
    # derivative of C w.r.t h
    JR[row_p:row_h, col_h:] =  diff_h_C[mesh.p_unknown]

# 4th block row of the Jacobian (K)
    # derivative of K w.r.t u
    JR[row_h:, col_u:col_w] = diff_u_K[mesh.surface_unknown][:, mesh.u_unknown]
    # derivative of K w.r.t w
    JR[row_h:, col_w:col_p] = diff_w_K[mesh.surface_unknown][:, mesh.w_unknown]
    # derivative of K w.r.t h
    JR[row_h:, col_h:] =  diff_h_K[mesh.surface_unknown]

    return JR