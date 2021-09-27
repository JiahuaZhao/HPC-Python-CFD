# 
# Jacobi routine for CFD calculation
#
import numpy as np
import numexpr as ne

import sys
import os
os.environ['NUMEXPR_MAX_THREADS'] = '128'
ne.set_num_threads(128)

def jacobistep(psi, m, n):
    """
    Generates one step of the jacobi function for the whole grid
    """
    #return 0.25 * (psi[0:m, 1:n+1]+psi[2:m+2, 1:n+1]+psi[1:m+1,  0:n] + psi[1:m+1, 2:n+2])
    return ne.evaluate("0.25 * (a + b + c + d)", {'a':psi[0:m, 1:n+1],'b':psi[2:m+2, 1:n+1],'c':psi[1:m+1, 0:n],'d':psi[1:m+1, 2:n+2]})

def jacobistepvort(zet, psi, m, n, re):
    #ne.set_num_threads(8)
    #ne.set_vml_num_threads(8)
    #print(np.sum(zet), np.sum(psi))
    #Re = re
    #psinew = 0.25 * (psi[0:m, 1:n+1]+psi[2:m+2, 1:n+1]+psi[1:m+1, 0:n] + psi[1:m+1, 2:n+2] - zet[1:m+1, 1:n+1])
    psinew = ne.evaluate("0.25 * (a + b + c + d - e)", {'a':psi[0:m, 1:n+1],'b':psi[2:m+2, 1:n+1],'c':psi[1:m+1, 0:n],'d':psi[1:m+1, 2:n+2],'e':zet[1:m+1, 1:n+1]})

    #zetnew = - re/16.0 * ((psi[1:m+1, 2:n+2]-psi[1:m+1, 0:n])*(zet[2:m+2, 1:n+1]-zet[0:m, 1:n+1]) - (psi[2:m+2, 1:n+1]-psi[0:m, 1:n+1])*(zet[1:m+1, 2:n+2]-zet[1:m+1, 0:n])) + (0.25*(zet[0:m, 1:n+1]+zet[2:m+2, 1:n+1]+zet[1:m+1, 0:n]+zet[1:m+1, 2:n+2]))
    zetnew = ne.evaluate("- re / 16.0 * ((d - c) * (f - g) - (b - a) * (h - i)) + (0.25 * (f + g + h + i))", {'re':re,'a':psi[0:m, 1:n+1],'b':psi[2:m+2, 1:n+1],'c':psi[1:m+1, 0:n],'d':psi[1:m+1, 2:n+2],'f':zet[2:m+2, 1:n+1],'g':zet[0:m, 1:n+1],'h':zet[1:m+1, 2:n+2],'i':zet[1:m+1, 0:n]})
    return psinew, zetnew


def deltasq(psi_os_zet_temp, oldarr, m, n):
    dsq = np.sum(np.power(psi_os_zet_temp - oldarr[1: m+1, 1:n+1], 2))
    return float(dsq)
