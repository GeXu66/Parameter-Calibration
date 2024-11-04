import numpy as np
import matplotlib.pyplot as plt


def box_con_ucon(lb, ub, dim):
    # lb is the input vector for the lower bounds
    # ub is the input vector for the upper bounds
    # z1 is the contraction scale factor, z2 is the contraction translation
    # z3 is the uncontraction scale factor, z4 is the uncontraction translation

    scale = np.zeros(dim)
    trans = np.zeros(dim)

    for i in range(dim):
        scale[i] = 1 / (ub[i] - lb[i])  # contraction scale factor
        trans[i] = -lb[i] / (ub[i] - lb[i])  # contraction translation

    z1 = scale
    z2 = trans
    z3 = 1. / scale  # uncontraction scale
    z4 = -trans * z3  # uncontraction translation

    return z1, z2, z3, z4


def coordinate_cons(x1, x2, dim):
    """
    x1: first extremum seed (numpy array)
    x2: second extremum seed (numpy array)
    dim: dimension (int)

    Returns a list of dictionaries with 'lb' (lower bound) and 'ub' (upper bound).
    """
    z = [{'lb': None, 'ub': None} for _ in range(dim)]  # Initialize a list of dictionaries for each dimension

    distt = np.linalg.norm(x1 - x2)  # compute the distance between x1 and x2
    length = distt / np.sqrt(dim)  # length of the hypercube
    c = (x1 + x2) / 2  # center of the hyper ball

    for i in range(dim):
        z[i]['lb'] = max(0, c[i] - length / 2)  # lower bound
        z[i]['ub'] = min(1, c[i] + length / 2)  # upper bound

    return z


import numpy as np


def find_neighbor(x, ObSet):
    """
    x: new sample (a dictionary or object with 'seed' attribute)
    ObSet: list of objects or dictionaries with 'seed' and 'level' attributes

    Returns the sample point with the extremum seed closest to the new extremum seed.
    """
    distt = []  # List to store distances
    sample_ind = []  # List to store indices of the extremum seeds
    num = 0  # Counter for valid extremum seeds at level 1

    # Iterate through ObSet to compute distances for level 1 extremum seeds
    for j in range(1, len(ObSet) - 1):
        if ObSet[j]['level'] == 1:  # Check if the extremum seed is at level 1
            distt.append(np.linalg.norm(x['seed'] - ObSet[j]['seed']))  # Compute Euclidean distance
            sample_ind.append(j)  # Store the index of the sample
            num += 1

    # If valid distances exist
    if distt:
        # Find the index of the minimum distance
        ind = np.argmin(distt)
        # Return the corresponding sample point with the minimum distance
        z = ObSet[sample_ind[ind]]
    else:
        # If no valid distances found, return the input sample
        z = x

    return z


def iterative_tunneling(x1, x2, L, globals):
    """
    x1: first sample point (dictionary/object with 'vector' and 'value')
    x2: second sample point (dictionary/object with 'vector' and 'value')
    L: level information of the input sample point
    globals: dictionary to store global variables like cs, tol1, tol2, ObSet, dim, CurrentBest, stopflg

    Returns the tunneling point at the Lth level.
    """
    # Extract global variables from the provided dictionary
    cs = globals['cs']
    tol1 = globals['tol1']
    tol2 = globals['tol2']
    ObSet = globals['ObSet']
    dim = globals['dim']
    CurrentBest = globals['CurrentBest']
    stopflg = globals['stopflg']

    # Stop if the flag indicates completion
    if stopflg[-1] == 1:
        return ObSet[-1]

    # Stop if L exceeds the square root of the dimension
    if L > np.sqrt(dim):
        return ObSet[-1]

    # Get the tunneling extremum seed (assuming tunneling is another function you will define)
    xt = tunneling(x1, x2, L)

    # Conditions for checking proximity to local minima
    if (np.linalg.norm((xt['vector'] * cs) - (x1['vector'] * cs)) / np.sqrt(dim) < tol1
            and abs(xt['value'] - x1['value']) < tol2):
        # No new local minimum found in the subdomain (do nothing)
        pass
    elif (np.linalg.norm((xt['vector'] * cs) - (x2['vector'] * cs)) / np.sqrt(dim) < tol1
          and abs(xt['value'] - x2['value']) < tol2):
        # No new local minimum found in the subdomain (do nothing)
        pass
    else:
        # Further tunneling based on value comparison
        if xt['value'] == CurrentBest['value']:
            # Exploitation: tunnel based on the smaller value between x1 and x2
            if x1['value'] < x2['value']:
                iterative_tunneling(CurrentBest, x1, L + 1, globals)
            else:
                iterative_tunneling(CurrentBest, x2, L + 1, globals)
        else:
            # Tunnel further between x1 and xt
            iterative_tunneling(x1, xt, L + 1, globals)

    return ObSet[-1]


def tunneling(x1, x2, L, globals):
    """
    x1: first sample point (dictionary/object with 'vector' and 'value')
    x2: second sample point (dictionary/object with 'vector' and 'value')
    L: level information of the input sample point
    globals: dictionary to store global variables

    Returns the tunneling extremum seed and stores it.
    """
    # Extract global variables from the provided dictionary
    ObSet = globals['ObSet']
    dim = globals['dim']
    stopflg = globals['stopflg']
    epsilon = globals['epsilon']
    delta = globals['delta']
    x_ini_tun = globals['x_ini_tun']
    cs = globals['cs']
    ct = globals['ct']
    Etrue = globals['Etrue']
    maxover = globals['maxover']
    CurrentBest = globals['CurrentBest']

    N = len(ObSet)

    # Check stop flag
    if stopflg[-1] == 1:
        return x1

    # Expected value too high
    if np.mean([x1['value'], x2['value']]) > np.mean(Etrue):
        return CurrentBest

    # Determine which point to use for the subdomain creation
    if x1['value'] <= x2['value']:
        Range = coordinate_cons(x1['vector'] * cs + ct, x2['vector'] * cs + ct, dim)  # Subdomain creation
    else:
        Range = coordinate_cons(x2['vector'] * cs + ct, x1['vector'] * cs + ct, dim)

    # Ensure the sampling is within the box (adjust bounds)
    for i in range(dim):
        if Range[i]['ub'] > 1:
            Range[i]['ub'] = 1
        if Range[i]['lb'] < 0:
            Range[i]['lb'] = 0

    # Sample a new point within the adjusted bounds
    xt = np.array([np.random.rand() * (Range[i]['ub'] - Range[i]['lb']) + Range[i]['lb'] for i in range(dim)])

    # Solve the optimization problem using the local method (assuming local_m is another function you will define)
    xo, eo, maxover = local_m(xt)

    # Store the tunneling extremum seed (assuming store is another function you will define)
    z = store(xt, xo, eo, L + 1)

    # Check stopping criteria (assuming check_stop is another function you will define)
    sfg = check_stop(eo, epsilon, delta)

    # Update the global variable x_ini_tun and stopflg
    globals['x_ini_tun'] = np.vstack([x_ini_tun, np.append(xt, L + 1)])
    globals['stopflg'] = np.append(stopflg, sfg)

    # Handle stopping conditions
    if sfg == 1 or maxover == 1:
        return x1

    return z


def store(x_ini, xo, eo, L, globals):
    """
    x_ini: extremum seed (numpy array)
    xo: design vector (numpy array)
    eo: function value of the design vector
    L: level information of the extremum seed
    globals: dictionary containing global variables

    Returns the newly stored local minimum.
    """
    # Extract global variables from the provided dictionary
    ObSet = globals['ObSet']
    EAP = globals['EAP']
    Su_MP = globals['Su_MP']
    W1t = globals['W1t']
    cs = globals['cs']
    ct = globals['ct']
    Etrue = globals['Etrue']
    CurrentBest = globals['CurrentBest']
    pop = globals['pop']
    num = globals['num']
    Ebest = globals['Ebest']
    nonlcon_i = globals['nonlcon_i']
    tol_eq_i = globals['tol_eq_i']
    num_consec_best = globals['num_consec_best']
    prvs_best = globals['prvs_best']
    num_consec_best_all = globals['num_consec_best_all']
    fmin = globals['fmin']
    fmax = globals['fmax']
    ee = globals.get('ee', 1)  # persistent variable initialized to 1

    # Check if x_ini and xo*cs + ct are not equal, and store into ObSet
    if np.linalg.norm(x_ini - (xo * cs + ct)) != 0:
        ObSet.append({
            'seed': x_ini,
            'vector': xo,
            'value': eo,
            'level': L,
            'radius': np.linalg.norm(x_ini - (xo * cs + ct))
        })

    # Check constraints if provided
    if nonlcon_i is not None:
        c, ceq = nonlcon_i(ObSet[-1]['vector'])
    else:
        c, ceq = 0, 0

    # Update CurrentBest if necessary (feasibility check)
    if CurrentBest['value'] > eo and np.all(c <= 0) and np.all(np.abs(ceq) <= tol_eq_i):
        CurrentBest = ObSet[-1]

    # Update consecutive best tracker
    if CurrentBest['value'] < prvs_best:
        num_consec_best = 0
        prvs_best = CurrentBest['value']
    else:
        num_consec_best += 1
        prvs_best = CurrentBest['value']

    num_consec_best_all.append(num_consec_best)

    # Update pop and energy variables
    pop[-1]['value'] = CurrentBest['value']
    pop[-1]['vector'] = CurrentBest['vector']
    Ebest[num] = CurrentBest['value']
    Etrue[num] = eo

    # Update fmin and fmax if necessary
    if eo < fmin:
        fmin = eo
    if eo > fmax:
        fmax = eo

    # Normalize and update EAP and Su_MP
    Np = normalize_up_to_t(eo)
    EAP.append((eo - 2 * min(0, fmin) + 1) / Np)

    if len(EAP) >= 2:
        w = np.exp(-min(EAP))
        Su_MP.append(w)
        if len(Su_MP) <= 1:
            W1t.append(1)
        else:
            W1t.append(1 - np.mean(Su_MP) / Su_MP[-2])

    # Update persistent variable
    globals['ee'] = ee

    # Return the last added element of ObSet
    return ObSet[-1]


if __name__ == "__main__":
    # 全局变量
    Su_MP = []
    MP = []
    CMP = []
    W1t = []
    Np = 0
    mt = []
    UpP = 0
    maxrun = 0
    alpha_tt = []
    fmin = 0
    Etrue_T = []
    plotflag = 0
    num = 0
    Up = []
    Etrue = []
    Ebest = []
    stop11 = []
    stop2a1 = []
