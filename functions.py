# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:04:24 2021

@author: Nicolai Almskou Rasmusen &
         Victor Mølbach Nissen
"""

# %% Imports

import numpy as np

# %% Functions


# ---- Modified ----
def delay_respons_vector(theta, tau, r, f, lambda_):
    # Angle
    e = np.matrix([np.cos(theta), np.sin(theta)])

    a = (np.exp(2j*np.pi*(1/lambda_) * e@r)).T

    # Delay
    b = (np.exp(-2j*np.pi*f*tau))

    # Return kronecker product
    return np.kron(b, a)


def addNoise(X, SNRdb):
    # Estiemate power
    x = np.matrix(X.flatten(order="F"))
    Pxx = (np.abs(x*np.conjugate(x.T))/x.size)[0, 0]

    # Calculate the variance
    Noise_power = (10**(-SNRdb/10))*Pxx

    # Calculate Noise
    W = np.sqrt(Noise_power/2)*(np.random.randn(X.shape[0], X.shape[1]) +
                                1j*np.random.randn(X.shape[0], X.shape[1]))

    return X + W


def estM(E, N, pn):
    p = np.arange(1, pn+1).reshape([1, pn])

    MMDL = N*np.log(E[np.arange(pn)])+(1/2)*(p**2 + p)*np.log(N)

    return np.argmin(MMDL) + 1


def getSubarray(Outer_dims, Inner_dims, offset=[0, 0, 0], spacing=1):
    """
    getSubarray gives you the index of the subarray of size L1 and L2 with
    respect to N_row and N_column.
    """

    idx_column = np.arange(0, Outer_dims[0], spacing
                           ).reshape([int(np.ceil(Outer_dims[0]/spacing)), 1])
    idx_row = np.arange(0, Outer_dims[1], spacing
                        ).reshape([int(np.ceil(Outer_dims[1]/spacing)), 1])
    idx_freq = np.arange(offset[2], offset[2]+Inner_dims[2], 1)

    if (len(idx_column) < Inner_dims[0]) or (len(idx_row) < Inner_dims[1]):
        print('Problem in finding the subarray')
        exit()
    else:
        idx_column = idx_column[offset[0]:offset[0]+Inner_dims[0]]

    idx_array = np.zeros([Inner_dims[0]*Inner_dims[1], 1], dtype=int)

    for il2 in range(Inner_dims[1]):
        idx_array[il2*Inner_dims[0]:(il2+1)*Inner_dims[0]] = \
            (idx_column + Outer_dims[0]*(il2+offset[1])*spacing)

    return [idx_array, idx_freq]


def spatialSmoothing(x, L, Ls, method=str):
    """
    L = [Lx, Ly, Lz]
    Ls = [Lsx, Lsy, Lsz]
    """
    # Split the signal array into P sub arays
    # And calculate the forward covariance
    Px, Py, Pz = L - Ls + 1

    RF = np.zeros([np.prod(Ls), np.prod(Ls)], dtype=np.complex128)

    for px in range(Px):
        for py in range(Py):
            for pz in range(Pz):
                idx_array = getSubarray(L, Ls, offset=[px, py, pz], spacing=1)

                xs = x[idx_array[0], idx_array[1]].flatten(order='F')
                xs = np.reshape(xs, (len(xs), 1))

                xsh = np.conjugate(xs).T

                RF += xs@xsh

    RF = RF/(Px*Py*Pz)

    # return forward
    if method == "forward":
        return RF

    # Backward Selection Matrix
    J_LS = np.flipud(np.eye(np.prod(Ls)))

    # Calculate forward-backward covariance
    return (1/2)*(RF+J_LS@np.conjugate(RF)@J_LS)


def MUSIC(R, Res, dat, idx_tau, idx_array, tau_search, M=None):
    # Parameters
    Tau = np.linspace(tau_search[0], tau_search[1], Res[1], endpoint=True)
    f = dat['f'][idx_tau]
    f0 = dat['f0']
    r = dat['r'][:, idx_array.reshape(len(idx_array))]

    lambda_ = 3e8/f0

    Theta = np.linspace(0, 2*np.pi, Res[0])
    Pm = np.zeros([Res[1], Res[0]])

    # ------ Step 3 - Form U ------
    E, U = np.linalg.eig(R)

    if M is None:
        M_est = estM(E, 101, 50)
    else:
        M_est = M
    print(M_est)
    Un = U[:, M_est:]

    # ------ Step 4 - Calculate Freq. estimate ------
    # Do the caluclations
    for i in range(Res[0]):
        print(i)
        for j in range(Res[1]):
            # Calculate for the different steering matrix
            As = delay_respons_vector(Theta[i], Tau[j], r, f, lambda_)

            Ash = np.conjugate(As).T
            Unh = np.conjugate(Un).T

            Pm[j, i] = 1/np.abs(Ash@Un@Unh@As)

            # print(f"step {i*len(Tau) + j + 1} out of {Res*len(Tau)}")

    return Pm


def bartlett(R, Res, dat, idx_tau, idx_array, tau_search):
    # Parameters
    Tau = np.linspace(tau_search[0], tau_search[1], Res[1], endpoint=True)
    f = dat['f'][idx_tau]
    f0 = dat['f0'][0, 0]
    r = dat['r'][:, idx_array.reshape(len(idx_array))]

    lambda_ = 3e8/f0

    Theta = np.linspace(0, 2*np.pi, Res[0])
    Pm = np.zeros([Res[1], Res[0]])

    # Do the caluclations
    for i in range(Res[0]):
        print(i)
        for j in range(Res[1]):
            As = np.zeros([len(f)*np.size(r, axis=1), 1], dtype=np.complex)
            Ash = np.zeros([1, len(f)*np.size(r, axis=1)], dtype=np.complex)

            # Calculate for the different steering matrix
            As = delay_respons_vector(Theta[i], Tau[j], r, f, lambda_)

            Ash = np.conjugate(As).T

            Pm[j, i] = np.abs(Ash@R@As)/(np.linalg.norm(As, ord=2)**4)

            # print(f"step {i*len(Tau) + j + 1} out of {Res*len(Tau)}")

    return Pm


def capon(R, Res, dat, idx_tau, idx_array, tau_search):
    # Parameters
    Tau = np.linspace(tau_search[0], tau_search[1], Res[1], endpoint=True)
    f = dat['f'][idx_tau]
    f0 = dat['f0']
    r = dat['r'][:, idx_array.reshape(len(idx_array))]

    lambda_ = 3e8/f0

    Theta = np.linspace(0, 2*np.pi, Res[0])
    Pm = np.zeros([Res[1], Res[0]])

    # Do the caluclations
    for i in range(Res[0]):
        print(i)
        for j in range(Res[1]):
            # Calculate for the different steering matrix
            As = delay_respons_vector(Theta[i], Tau[j], r, f, lambda_)

            Ash = np.conjugate(As).T

            Rinv = np.linalg.inv(R)

            Pm[j, i] = 1/np.abs(Ash@Rinv@As)

    return Pm
