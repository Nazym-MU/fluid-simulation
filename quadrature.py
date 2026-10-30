import numpy as np

def quadrature():

    # Gaussian quadrature points
    V1 = np.array([-1, -1])
    V2 = np.array([1, -1])
    V3 = np.array([-1, 1])

    n_Gaussian_Q = 16
    xi_eta_G = np.zeros((16, 2))
    W_G = np.zeros(16) # weights

    xi_eta_G[0, :] = V1/3 + V2/3 + V3/3
    W_G[0] = .1443156076777871682510911104890646

    a = [.1705693077517602066222935014914645, .0505472283170309754584235505965989, .4592925882927231560288155144941693]
    p = [.1032173705347182502817915502921290, .0324584976231980803109259283417806, .0950916342672846247938961043885843]
    kk = 0
    for ii in range(3):
        W_G[kk+1:kk+4] = p[ii]
        kk += 1
        xi_eta_G[kk, :] =       a[ii]*V1 +       a[ii]*V2 + (1-2*a[ii])*V3
        kk += 1
        xi_eta_G[kk, :] =       a[ii]*V1 + (1-2*a[ii])*V2 +       a[ii]*V3
        kk += 1
        xi_eta_G[kk, :] = (1 - 2 * a[ii])*V1 +       a[ii]*V2 +       a[ii]*V3

    b = .2631128296346381134217857862846436
    c = .0083947774099576053372138345392944

    W_G[kk+1:kk+7] = .0272303141744349942648446900739089
    kk = kk+1
    xi_eta_G[kk, :] =       b*V1 +       c*V2 + (1-b-c)*V3
    kk = kk+1
    xi_eta_G[kk, :] =       b*V1 + (1-b-c)*V2 +       c*V3
    kk = kk+1
    xi_eta_G[kk, :] =       c*V1 +       b*V2 + (1-b-c)*V3
    kk = kk+1
    xi_eta_G[kk, :] =       c*V1 + (1-b-c)*V2 +       b*V3
    kk = kk+1
    xi_eta_G[kk, :] = (1-b-c)*V1 +       b*V2 +       c*V3
    kk = kk+1
    xi_eta_G[kk, :] = (1-b-c)*V1 +       c*V2 +       b*V3

    xi_G = xi_eta_G[:, 0]
    eta_G = xi_eta_G[:, 1]
    W_G = 2*W_G # the weights are for a unit area triangle, so they need to be multiplied by the area of the triangle

    T = np.zeros((n_Gaussian_Q, 6, 6))
    phi_xi = np.zeros((6,n_Gaussian_Q))
    phi_eta = np.zeros((6,n_Gaussian_Q))
    for pp in range(n_Gaussian_Q):
        phi_xi[0, pp] = 1/2 + xi_G[pp] + eta_G[pp]
        phi_xi[1, pp] = 1/2 + xi_G[pp]
        phi_xi[2, pp] = 0
        phi_xi[3, pp] = -1 - eta_G[pp]
        phi_xi[4, pp] = -1 - eta_G[pp] - 2 * xi_G[pp]
        phi_xi[5, pp] = 1 + eta_G[pp]
        phi_eta[0, pp] = 1/2 + xi_G[pp] + eta_G[pp] 
        phi_eta[1, pp] = 0
        phi_eta[2, pp] = 1/2 + eta_G[pp] 
        phi_eta[3, pp] = -1 - xi_G[pp] - 2 * eta_G[pp]
        phi_eta[4, pp] = -1 - xi_G[pp]
        phi_eta[5, pp] = 1 + xi_G[pp]

        for ii in range(6):
            for jj in range(6):
                T[pp, ii, jj] = phi_xi[ii, pp] * phi_eta[jj, pp] -phi_eta[ii, pp] * phi_xi[jj, pp]

    # Matrix psi_G stores the values of psi_i in each of the Guassian integration points
    psi_G = np.zeros((3,n_Gaussian_Q))
    for pp in range(n_Gaussian_Q):
        psi_G[0, pp] = -(xi_G[pp] + eta_G[pp]) / 2 
    for pp in range(n_Gaussian_Q):
        psi_G[1, pp] = (1 + xi_G[pp]) / 2
    for pp in range(n_Gaussian_Q):
        psi_G[2, pp] = (1 + eta_G[pp]) / 2

    # Matrix phi_G stores the values of phi_i in each of the Guassian integration points
    phi_G = np.zeros((6,n_Gaussian_Q))
    for jj in range(3):
        for pp in range(n_Gaussian_Q):
            phi_G[jj,pp] = psi_G[jj,pp] * (2 * psi_G[jj,pp] - 1)
    for pp in range(n_Gaussian_Q):
        phi_G[3,pp] = 4 * psi_G[0,pp] * psi_G[2,pp]
    for jj in range(4, 6):
        for pp in range(n_Gaussian_Q):
            phi_G[jj,pp] = 4*psi_G[jj-3,pp]*psi_G[jj-4,pp]

    # Matrices needed for line-element integration
    xi_lG = np.array([-0.9782286581460569928039, -0.8870625997680952990752,
             -0.7301520055740493240934, -0.5190961292068118159257,
             -0.2695431559523449723315, 0, 0.269543155952344972332,
             0.5190961292068118159257, 0.7301520055740493240934,
             0.887062599768095299075, 0.9782286581460569928039])
    n_lGaussian_Q = len(xi_lG)
    
    W_lG = np.array([0.0556685671161736664828, 0.1255803694649046246347, 
            0.1862902109277342514261, 0.2331937645919904799185,
            0.2628045445102466621807, 0.2729250867779006307145,
            0.262804544510246662181, 0.2331937645919904799185,
            0.1862902109277342514261, 0.1255803694649046246347,
            0.055668567116173666483])
    
    # assesing the line-element hat functions at the Gaussian quadrature points for boundary 1
    phi1_lG = np.zeros((3, n_lGaussian_Q))
    for pp in range(n_lGaussian_Q):
        phi1_lG[0, pp] = xi_lG[pp]**2 / 2 - xi_lG[pp] / 2
        phi1_lG[1, pp] = 1-xi_lG[pp]**2
        phi1_lG[2, pp] = xi_lG[pp]**2 / 2 + xi_lG[pp]/2

    phi1_xi_lG = np.zeros((3, n_lGaussian_Q))
    for pp in range(n_lGaussian_Q):
        phi1_xi_lG[0, pp] = xi_lG[pp] - 1/2
        phi1_xi_lG[1, pp] = -2*xi_lG[pp]
        phi1_xi_lG[2, pp] = xi_lG[pp] + 1/2

    return  {'xi_G': xi_G,
             'eta_G': eta_G,
             'W_G': W_G,
             'phi_G': phi_G,
             'psi_G': psi_G,
             'phi_xi': phi_xi,
             'phi_eta': phi_eta,
             'T': T,
             'xi_lG': xi_lG,
             'W_lG': W_lG,
             'phi1_lG': phi1_lG,
             'phi1_xi_lG': phi1_xi_lG }