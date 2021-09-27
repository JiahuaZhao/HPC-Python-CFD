#!/usr/bin/env python
#
# CFD Calculation
# ===============
#
# Simulation of inviscid flow in a 2D box using the Jacobi algorithm.
#
# Python version - uses numpy and loops
#
# EPCC, 2014
#
import sys
import time

# Import numpy
import numpy as np
from copy import deepcopy

# Import the local "util.py" methods
import util

# Import the external jacobi function from "jacobi.py"
from jacobi import jacobistepvort, deltasq, jacobistep


def main(argv):
    # Test we have the correct number of arguments
    if len(argv) < 2:
        sys.stdout.write("Usage: cfd.py <scalefactor> <iterations> [reynolds]\n")
        sys.exit(1)

    # Get the systen parameters from the arguments
    scalefactor = int(argv[0])
    niter = int(argv[1])

    sys.stdout.write("\n2D CFD Simulation\n")
    sys.stdout.write("=================\n")
    sys.stdout.write("Scale factor = {0}\n".format(scalefactor))
    sys.stdout.write("Iterations   = {0}\n".format(niter))

    # check command line and add reynolds
    if len(argv) == 3:
        re = float(argv[2])
        irrotational = 0
        print(f"Reynolds number = {re}")
    else:
        re = -1
        irrotational = 1
        print("Irrotational flow\n")

    # print interval
    printfreq = 1000
    # Set the minimum size parameters
    bbase = 10
    hbase = 15
    wbase = 5
    mbase = 32
    nbase = 32

    # Set the parameters for boundary conditions
    b = bbase * scalefactor
    h = hbase * scalefactor
    w = wbase * scalefactor
    # Set the dimensions of the array
    m = mbase * scalefactor
    n = nbase * scalefactor

    # reynolds number
    re = re / scalefactor
    # irrotational?
    if not irrotational:
        zet = np.zeros((m + 2, n + 2))
    # checkreynolds
    checkerr = 0
    # //tolerance for convergence. <=0 means do not check
    tolerance = 0

    #  //do we stop because of tolerance?
    if tolerance > 0:
        checkerr = 1

    # Write the simulation details
    sys.stdout.write("\nGrid size = {0} x {1}\n".format(m, n))

    # Define the psi array of dimension [m+2][n+2] and set it to zero
    psi = np.zeros((m + 2, n + 2))
    print('psi', psi)
    # Set the boundary conditions on bottom edge
    for i in range(b + 1, b + w):
        psi[i][0] = float(i - b)
    for i in range(b + w, m + 1):
        psi[i][0] = float(w)

    # Set the boundary conditions on right edge
    for j in range(1, h + 1):
        psi[m + 1][j] = float(w)
    for j in range(h + 1, h + w):
        psi[m + 1][j] = float(w - j + h)

    # compute normalisation factor for error
    # iterative
    bnorm = 0
    for i in range(m+2):
        for j in range(n+2):
            bnorm += psi[i][j]*psi[i][j]
    
    if not irrotational:
        # update zeta BCs that depends on psi
        _boundaryzet(zet, psi, m, n)

        # iterative:
        for i in range(m + 2):
            for j in range(n + 2):
                bnorm += zet[i][j] * zet[i][j]

    bnorm = np.sqrt(bnorm)

    # Call the Jacobi iterative loop (and calculate timings)
    sys.stdout.write("\nStarting main Jacobi loop ...\n\n")
    tstart = time.time()
    # OLD:: jacobi(niter, psi)
    # -------------------
    for iter in range(1, niter + 1):
        # //calculate psi for next iteration
        if irrotational:
            psitmp = jacobistep(psi, m, n)
        else:
            psitmp, zettmp = jacobistepvort(zet, psi, m, n, re)

        # //calculate current error if required
        if checkerr or iter == niter:
            error = deltasq(psitmp, psi, m, n)

            if not irrotational:
                error += deltasq(zettmp, zet, m, n)
            error = np.sqrt(error) / bnorm

        # //copy back but not all!!
        # iterative version:
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                psi[i][j] = psitmp[i][j]

        if not irrotational:
            # //copy back but not all!!
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    zet[i][j] = zettmp[i][j]
            #zet[1:m+1][1:n+1] = zettmp[1:m+1][1:n+1]

            # update zeta BCs that depend on psi
            _boundaryzet(zet, psi, m, n)

        #       //quit early if we have reached required tolerance
        if checkerr and error < tolerance:
            print(f"Converged on iteration {iter}")
            break

        # //print loop information
        if (iter % printfreq == 0):
            if not checkerr:
                print(f"Completed iteration {iter}")
            else:
                print(f"Completed iteration {iter}, error = {error}\n")

    if iter > niter:
        iter = niter
    # -------------------

    tend = time.time()
    sys.stdout.write("\n... finished\n")
    sys.stdout.write("\nCalculation took {0:.5f}s\n\n".format(tend - tstart))

    ttot = tend - tstart
    titer = ttot / iter
    # print out some stats
    print("\n... finished\n")
    print(f"After {iter}  iterations, the error is {error}\n")
    print(f"Time for {iter} iterations was {ttot} seconds\n")
    print(f"Each iteration took {titer} seconds\n")

    print(scalefactor)
    # Write the output files for subsequent visualisation
    #util.write_data(m, n, scalefactor, psi, "velocity.dat", "colourmap.dat")

    # generate gnuplot file
    #util.writeplotfile(m, n, scalefactor)

    # Finish nicely
    sys.exit(0)


def _boundaryzet(zet, psi, m, n):
    # set top/bottom BCs:
    for i in range(1, m+1):
        zet[i][0]   = 2*(psi[i][1]-psi[i][0])
        zet[i][n+1] = 2*(psi[i][n]-psi[i][n+1])
    
    # set left BCs:
    for j in range(1, n+1):
        zet[0][j] = 2.0*(psi[1][j]-psi[0][j])

    # set right BCs
    for j in range(1, n+1):
        zet[m+1][j] = 2.0*(psi[m][j]-psi[m+1][j])


# Function to create tidy way to have main method
if __name__ == "__main__":
    main(sys.argv[1:])
