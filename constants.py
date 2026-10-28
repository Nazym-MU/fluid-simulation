def make_constants(length, height, n_spines,  layers):
    '''
    Inputs:
    - layers: the number of rectangular cells in a single column; one rectangular cell = 2 triangles
    - n_spines: the number of vertical spines (must be an odd number)

    Returns:
    - A dictionary of parameters. Keys ending in '_dim' are dimensional and in SI units. All other keys are dimensionless.
    '''
    # Dimensional constants

    g_dim = 9.8 # m/s^2
    rho_dim = 1000 # kg/m^3
    nu_dim = 1e-6 # m^2/s
    sigma_dim = 72.86e-3 #Kg/s^2 (i.e. N/m)

    # fluid bath width and depth
    L_dim = length # m
    H_dim = height # m

    U_dim = sigma_dim / (rho_dim * nu_dim)
    T_dim = L_dim / U_dim
    P_dim = rho_dim * nu_dim * U_dim / L_dim

    # dimensionless variables
    Re =  U_dim * L_dim / nu_dim # Reynolds number
    St = g_dim * L_dim**2 / (nu_dim * U_dim) # Stokes number
    Ca = rho_dim * nu_dim * U_dim / sigma_dim # Capillary number

    L = L_dim / L_dim
    H = H_dim / L_dim


    constants = {'g_dim': g_dim,
                 'rho_dim': rho_dim,
                 'nu_dim': nu_dim,
                 'sigma_dim': sigma_dim,
                 'L_dim': L_dim,
                 'H_dim': H_dim,
                 'L': L,
                 'H': H, 
                 'n_spines': n_spines,
                 'layers': layers,
                 'U_dim': U_dim,
                 'T_dim': T_dim,
                 'P_dim': P_dim,
                 'Re': Re,
                 'St': St,
                 'Ca': Ca}
    return constants