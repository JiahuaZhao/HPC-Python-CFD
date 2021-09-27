# 
# Jacobi routine for CFD calculation
#
import numpy as np

import sys

def jacobistep(psi, m, n):
    """
    Generates one step of the jacobi function for the whole grid
    """
    return 0.25 * (psi[0:m, 1:n+1]+psi[2:m+2, 1:n+1]+psi[1:m+1,  0:n] + psi[1:m+1, 2:n+2])


def jacobistepvort(zet, psi, m, n, re):
    #print(np.sum(zet), np.sum(psi))
    psinew = 0.25 * (psi[0:m, 1:n+1]+psi[2:m+2, 1:n+1]+psi[1:m+1, 0:n] + psi[1:m+1, 2:n+2] - zet[1:m+1, 1:n+1])

    zetnew = - re/16.0 * ((psi[1:m+1, 2:n+2]-psi[1:m+1, 0:n])*(zet[2:m+2, 1:n+1]-zet[0:m, 1:n+1]) - (psi[2:m+2, 1:n+1]-psi[0:m, 1:n+1])*(zet[1:m+1, 2:n+2]-zet[1:m+1, 0:n])) + (0.25*(zet[0:m, 1:n+1]+zet[2:m+2, 1:n+1]+zet[1:m+1, 0:n]+zet[1:m+1, 2:n+2]))
    return psinew, zetnew


def deltasq(psi_os_zet_temp, oldarr, m, n):
    dsq = np.sum(np.power(psi_os_zet_temp - oldarr[1: m+1, 1:n+1], 2))
    return float(dsq)
