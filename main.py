# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:04:24 2021

@author: Nicolai Almskou Rasmusen &
         Victor Mølbach Nissen
"""

# %% Imports

import numpy as np
import scipy.io as scio
import functions as fun

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'

# %% functions


# %% Main

if __name__ == '__main__':
    # ---- Parameters ----
    Res = [100, 100]  # Res for [theta, tau]

    # SNR in [dB]
    SNRdb = -10

    # Number of sources
    M = 5

    # Matrix dim. of given data
    N_row = 71
    N_column = 66
    freq_samples = 101

    # For: getSubarray
    L1 = 15  # Number of sub rows
    L2 = 15  # Number of sub columns
    L3 = 20  # Number of sub samples

    # Get the index for the subarrays
    array_size = np.array([N_row, N_column, freq_samples])
    subarray_size = np.array([L1, L2, L3])
    smoothing_array_size = np.array([6, 6, 10])

    # Search Space
    tau_search = [0, 5e-7]

    # plot
    plot = 1

    # ---- Initialise data ----
    # Load datafile
    dat = scio.loadmat("data.mat")

    # Freq. domain
    X = dat['X_synthetic']
    # X = dat['X']

    # ---- Add noise ----
    X = fun.addNoise(X, SNRdb)

    # Index data vector for antennas in subarray
    idx_array = fun.getSubarray(array_size, subarray_size,
                                offset=[0, 0, 0], spacing=2)

    # We need a L*Lf vector. Need to flatten it columnmajor (Fortran)
    X_sub = X[idx_array[0], idx_array[1]]

    # ----- Spatial Smoothing -----
    print("smooth start")
    RFB = fun.spatialSmoothing(X_sub,
                               subarray_size,
                               smoothing_array_size)

    # Need to use spatial smoothing when using MUSIC as rank is 1
    X_sub = X_sub.flatten(order='F').reshape(
        len(X_sub.flatten(order='F')), 1, order='F')
    R = X_sub @ (np.conjugate(X_sub).T)

    # Do the Algorithms
    print("Algorithms")
    idx_subarray = fun.getSubarray(array_size, smoothing_array_size,
                                   offset=[0, 0, 0], spacing=2)
    print("Capon")
    Pm_Capon = fun.capon(RFB, Res, dat, idx_subarray[1],
                         idx_subarray[0], tau_search)

    print("Bartlett")
    Pm_Bartlett = fun.bartlett(R, Res, dat, idx_array[1],
                               idx_array[0], tau_search)

    print("MUSIC")
    Pm_MUSIC = fun.MUSIC(RFB, Res, dat, idx_subarray[1],
                         idx_subarray[0], tau_search, M=M)

    # %% Plot
    AoA = (dat['smc_param'][0][0][1])*180/np.pi
    AoA[AoA < 0] = AoA[AoA < 0] + 360

    TDoA = (dat['smc_param'][0][0][2])*(1/3e8) + np.abs(dat['tau'][0])

    if plot == 1:

        plt.figure()
        plt.title(f"Capon- Sweep - res: {Res} - SNRdb: {SNRdb}")
        plt.scatter(AoA, TDoA, color='r', marker='x')
        pm_max = np.max(10*np.log10(Pm_Capon))
        # pm_max = 20
        plt.imshow(10*np.log10(Pm_Capon), vmin=pm_max-40, vmax=pm_max,
                   extent=[0, 360,
                           tau_search[0], tau_search[1]],
                   aspect="auto", origin="lower")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [degrees]")

        plt.figure()
        plt.title(f"Bartlett- Sweep - res: {Res} - SNRdb: {SNRdb}")
        plt.scatter(AoA, TDoA, color='r', marker='x')
        pm_max = np.max(10*np.log10(Pm_Bartlett))
        # pm_max = 20
        plt.imshow(10*np.log10(Pm_Bartlett), vmin=pm_max-40, vmax=pm_max,
                   extent=[0, 360,
                           tau_search[0], tau_search[1]],
                   aspect="auto", origin="lower")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [degrees]")

        plt.figure()
        plt.title(f"MUSIC- Sweep - res: {Res} - SNRdb: {SNRdb}")
        plt.scatter(AoA, TDoA, color='r', marker='x')
        pm_max = np.max(10*np.log10(Pm_MUSIC))
        # pm_max = 20
        plt.imshow(10*np.log10(Pm_MUSIC), vmin=pm_max-40, vmax=pm_max,
                   extent=[0, 360,
                           tau_search[0], tau_search[1]],
                   aspect="auto", origin="lower")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [degrees]")

    elif plot == 2:
        x = np.linspace(-180, 180, Res[0])
        y = np.linspace(tau_search[0], tau_search[1], Res[1], endpoint=True)
        z = 10*np.log10(Pm_MUSIC)

        x, y = np.meshgrid(x, y)
        fig2 = go.Figure(data=[go.Surface(z=z, x=x,
                                          y=y)])

        fig2.update_layout(scene=dict(
            xaxis_title='Azimuth Angle - degrees',
            yaxis_title='Delay - micro-seconds'),
            title=f"Sweep - res: {Res}X{Res} points",
        )

        fig2.show()

    print("Hello World")  # Prints "Hello World"


# %%
if plot == 3:
    from matplotlib import cm
    x = np.linspace(-180, 180, Res[0])
    y = np.linspace(tau_search[0], tau_search[1], Res[1], endpoint=True)
    z = 10*np.log10(Pm_Bartlett)

    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    plt.ylabel("Tau [s]")
    plt.xlabel("Theta [degrees]")
    plt.show()
