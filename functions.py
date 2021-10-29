# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:04:24 2021

@author: Nicolai Almskou Rasmusen &
         Victor Mølbach Nissen
"""

# %% Imports

import numpy as np

from scipy.signal import find_peaks

# %% Functions


# ---- Modified ----
def delay_respons_vector(theta, tau, r, f, lambda_):

    # Parameters
    f0 = f[0]  # Not the carrier freq. but the first freq. in sweep
    df = f[1]-f[0]

    # Angle
    a = (np.exp(-1j*2*np.pi*(1/lambda_) *
         np.array([np.cos(theta), np.sin(theta)]).T@r))

    # Delay
    """
    b = (np.exp(-1j*2*np.pi*np.arange(0, len(f))*tau*df) *
         np.exp(-1j*2*np.pi*f0*tau))
    """

    b = (np.exp(-1j*2*np.pi*f*tau)).reshape(len(f))

    # Return kronecker product
    return np.kron(a, b).T


def MUSIC(R, Res, dat, idx_tau, idx_array, M):
    # Parameters
    Tau = dat['tau']
    f = dat['f'][idx_tau]
    f0 = dat['f0']
    r = dat['r'][:, idx_array.reshape(len(idx_array))]

    lambda_ = 3e8/f0

    Theta = np.linspace(0, 2*np.pi, Res)
    Pm = np.zeros([Res, len(Tau)])

    # ------ Step 3 - Form U ------
    E, U = np.linalg.eig(R)
    Un = U[:, M:]

    # ------ Step 4 - Calculate Freq. estimate ------
    # Do the caluclations
    for i in range(len(Theta)):
        for j in range(len(Tau)):
            # Calculate for the different steering matrix
            As = delay_respons_vector(Theta[i], Tau[j], r, f, lambda_)

            Ash = np.conjugate(As).T
            Unh = np.conjugate(Un).T

            Pm[i, j] = 1/np.abs(Ash@Un@Unh@As)

            print(f"step {i*len(Tau) + j + 1} out of {Res*len(Tau)}")

    return Pm


def getSubarray(N_row, N_column, L1, L2, spacing=1):
    """
    getSubarray gives you the index of the subarray of size L1 and L2 with
    respect to N_row and N_column.
    """

    idx_column = np.arange(0, N_row, spacing
                           ).reshape([int(N_row/spacing), 1])
    idx_row = np.arange(0, N_column, spacing
                        ).reshape([int(N_column/spacing), 1])

    if (len(idx_column) < L1) or (len(idx_row) < L2):
        print('Problem in finding the subarray')
        exit()
    else:
        idx_column = idx_column[0:L1]
        idx_row = idx_row[0:L2]

    idx_array = np.zeros([L1*L2, 1], dtype=int)

    for il2 in range(L2):
        idx_array[il2*L1:(il2+1)*L1] = (idx_column +
                                        N_row*(il2)*spacing)

    return idx_array


def spatialSmoothing(x, L, Ls, method=str):
    """
    L = [Lx, Ly, Lz]
    Ls = [Lsx, Lsy, Lsz]
    """
    # Split the signal array into P sub arays
    # And calculate the forward covariance
    x_cube = x.reshape(L, order='F')

    Px, Py, Pz = L - Ls + 1

    RF = np.zeros([np.prod(Ls), np.prod(Ls)], dtype=np.complex128)

    for px in range(Px):
        for py in range(Py):
            for pz in range(Pz):
                xs = x_cube[px:(px+Ls[0]),
                            py:(py+Ls[1]),
                            pz:(pz+Ls[2])].flatten(order='F')
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


def barlettRA(X, Res, dat, idx_tau, idx_array):
    # Parameters
    Tau = dat['tau']
    f0 = dat['f0']
    r = dat['r'][:, idx_array.reshape(len(idx_array))]

    aoa_search = np.linspace(0, 2*np.pi, Res)
    DTFT_aoa = np.zeros([Res, len(idx_array)], dtype=np.complex128)

    for im in range(Res):
        DTFT_aoa[im, :] = np.exp(-1j*2*np.pi*(f0/3e8) * np.array(
                                 [np.cos(aoa_search[im]),
                                  np.sin(aoa_search[im])]).T@r)

    K = 300
    tau_search = np.linspace(0, 1, K)
    f_tau = np.arange(0, len(Tau))

    DTFTconj_delay = np.zeros([len(Tau), K], dtype=np.complex128)

    for ik in range(K):
        DTFTconj_delay[:, ik] = np.exp(1j*2*np.pi*f_tau*tau_search[ik])

    return np.abs(DTFT_aoa@X@DTFTconj_delay)


def barlett(R, Res, dat, idx_tau, idx_array):
    # Parameters
    Tau = dat['tau']
    f = dat['f'][idx_tau]
    f0 = dat['f0']
    r = dat['r'][:, idx_array.reshape(len(idx_array))]

    lambda_ = 3e8/f0

    Theta = np.linspace(0, np.pi, Res)
    Pm = np.zeros([Res, len(Tau)])

    # Do the caluclations
    for i in range(len(Theta)):
        for j in range(len(Tau)):
            # Calculate for the different steering matrix
            As = delay_respons_vector(Theta[i], Tau[j], r, f, lambda_)

            Ash = np.conjugate(As).T

            Pm[i, j] = np.abs(Ash@R@As)/(np.linalg.norm(As)**4)

            print(f"step {i*len(Tau) + j + 1} out of {Res*len(Tau)}")

    return Pm


def capon(R, Res, dat, idx_tau, idx_array):
    # Parameters
    Tau = dat['tau']
    f = dat['f'][idx_tau]
    f0 = dat['f0']
    r = dat['r'][:, idx_array.reshape(len(idx_array))]

    lambda_ = 3e8/f0

    Theta = np.linspace(0, np.pi, Res)
    Pm = np.zeros([Res, len(Tau)])

    # Do the caluclations
    for i in range(len(Theta)):
        for j in range(len(Tau)):
            # Calculate for the different steering matrix
            As = delay_respons_vector(Theta[i], Tau[j], r, f, lambda_)

            Ash = np.conjugate(As).T

            Rinv = np.linalg.inv(R)

            Pm[i, j] = 1/np.abs(Ash@Rinv@As)

            print(f"step {i*len(Tau) + j + 1} out of {Res*len(Tau)}")

    return Pm

# ---- Old ----
def steering_matrix_1d(L, theta, d, lambda_):
    return np.exp(1j*2*np.pi*np.arange(0, L).reshape(L, 1) *
                  (d/lambda_)*np.cos(theta))


def steering_matrix_2d(L2d, AoA, d, lambda_):
    theta, phi = AoA
    Lx, Ly = L2d
    dx, dy = d

    eta_x = (dx/lambda_)*np.sin(theta)*np.cos(phi).reshape(1, np.size(theta))
    eta_y = (dy/lambda_)*np.sin(theta)*np.sin(phi).reshape(1, np.size(theta))

    A = []

    for i in range(Lx):
        A.append(np.exp(1j*2*np.pi*(
            np.ones([Lx, 1])*i*eta_x +
            np.arange(0, Ly).reshape(Ly, 1)*eta_y)))

    return np.concatenate(A)


def get_peaks(Pm, Res, M):
    Theta = np.linspace(0, np.pi, Res)*(180/np.pi)

    peaks, _ = find_peaks(Pm.reshape(Res))
    tmp = Pm[peaks]
    tmp_sort = np.argsort(tmp, axis=0)

    return Theta[peaks[tmp_sort[-M:]]].reshape([np.min([M, len(tmp_sort)])])


def DMLE(theta, x, L, M):
    R = np.cov(x, bias=True)

    A = steering_matrix_1d(L, theta)
    AT = np.linalg.pinv(A)
    A_ort = np.eye(L)-A@AT

    return (1/(L-M))*np.trace(A_ort@R)


def ESPRIT(R, Res, M, L2d):

    E, U = np.linalg.eig(R)
    Us = U[:, :M]

    # Calculate selections matrices
    J1 = np.eye(L2d[0])[:-1, :]
    J2 = np.eye(L2d[0])[1:, :]

    Ju1 = np.kron(np.eye(L2d[1]), J1)
    Ju2 = np.kron(np.eye(L2d[1]), J2)

    Jv1 = np.kron(J1, np.eye(L2d[1]))
    Jv2 = np.kron(J2, np.eye(L2d[1]))

    # ------ Step 4 - Calculate C ------
    # Calculate subarrays
    phi_u = np.linalg.pinv(Ju1@Us)@Ju2@Us
    phi_v = np.linalg.pinv(Jv1@Us)@Jv2@Us

    # ------ Step 5 - Calculate Eigenvalues ------
    Eu, Uu = np.linalg.eig(phi_u)
    Ev, Uv = np.linalg.eig(phi_v)

    u = (1/np.pi)*np.angle(Eu)
    v = (1/np.pi)*np.angle(Ev)

    theta = np.angle(u+1j*v)
    phi = np.arcsin(np.abs(u+1j*v))

    return theta, phi
