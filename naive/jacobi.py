# 
# Jacobi function for CFD calculation
#
# Basic Python version using lists
#

# Jacobi routine for CFD calculation
#
import numpy as np

import sys



def jacobistep(psi, m, n):
    """
    Generates one step of the jacobi function for the whole grid
    """
    psinew = np.zeros((m + 2, n + 2))
    # iterative version
    for i in range(1, m+1):
        for j in range(1, n+1):
            psinew[i][j]=0.25*(psi[i-1][j] +psi[i+1][j] + psi[i][j-1] + psi[i][j+1])

    return psinew


def jacobistepvort(zet, psi, m, n, re):

    zetnew = np.zeros((m + 2, n + 2))
    psinew = np.zeros((m + 2, n + 2))

    # iterative:
    for i in range(1, m+1):
        for j in range(1, n+1):
        # psi
            psinew[i][j]=0.25*(psi[i-1][j]+psi[i+1][j]+psi[i][j-1]+psi[i][j+1] - zet[i][j])
    # z
    # iterative::
    for i in range(1, m+1):
        for j in range(1, m+1):
            zetnew[i][j]=0.25*(zet[i-1][j]+zet[i+1][j]+zet[i][j-1]+zet[i][j+1]) - re/16.0* ((psi[i][j+1]-psi[i][j-1])*(zet[i+1][j]-zet[i-1][j]) - (psi[i+1][j]-psi[i-1][j])*(zet[i][j+1]-zet[i][j-1]))
    # print(psinew)
    return psinew, zetnew


def deltasq(newarr, oldarr, m, n):
    dsq = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dsq += (newarr[i][j] - oldarr[i][j]) ** 2
    return float(dsq)
