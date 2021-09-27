

#!/usr/bin/env python
#
# CFD Calculation with MPI4PY
# ===============
#
# Simulation of inviscid flow in a 2D box using the Jacobi algorithm.
#
# Python version - uses numpy and loops
#
# Alejandro Dinkelberg
#
import os
import sys
#import mkl
import time
import mpi4py.MPI as MPI

# Import numpy
import numpy as np
import numexpr as ne
from copy import deepcopy
os.environ['NUMEXPR_MAX_THREADS'] = '128'
ne.set_num_threads(2)
#mkl.set_num_threads(128)
#ne.set_vml_num_threads(128)
#ne.set_vml_accuracy_mode('fast')

##################################################################################################################################################################
# boundary and haloSWAP

def boundarypsi(psi, m, n, b, h, w, comm):
    # initialize the std values MPI
    rank = comm.Get_rank()
    size = comm.Get_size()

    istart = m*rank + 1
    istop = istart + m - 1
    
    # BCs on bottom edge
    for i in range(b+1, b+w):
        if i >= istart and i <= istop:
            psi[i-istart+1][0] = i-b

    for i in range(b+w, m*size+1):
        if i >= istart and i <= istop:
            psi[i-istart+1][0] = w
    
    # BCS on RHS
    if rank == size-1:
      for j in range(1, h+1):
          psi[m+1][j] = w
      for j in range(h+1, h+w):
          psi[m+1][j]= w-j+h


def boundaryzet(zet, psi, m, n, comm):
    # initialize the std values MPI
    rank = comm.Get_rank()
    size = comm.Get_size()

    istart = m*rank + 1
    istop = istart + m - 1

    # set top/bottom BCs:
    zet[1:m+1, 0] = 2 * (psi[1:m+1, 1] - psi[1:m+1, 0])
    zet[1:m+1, n+1] = 2 * (psi[1:m+1, n] - psi[1:m+1, n+1])

    # Set left BCs
    if 0 == rank:
        zet[0, 1:n+1] = 2 * (psi[1, 1:n+1] - psi[0, 1:n+1])

    # Set right BCs
    if size-1 == rank:
        zet[m+1, 1:n+1] = 2 * (psi[m, 1:n+1] - psi[m+1, 1:n+1])

    return zet

def haloSWAP(x, lm, n, comm):
    tag = 1
    status = MPI.Status()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # no need to halo swap if serial:
    if size > 1:
        # send right boundaries and receive left ones
        if rank == 0:
            comm.Send(x[lm][1:n+1], rank+1, tag)
        elif rank == size-1:
            comm.Recv(x[0][1:n+1], rank-1, tag, status)
        else:
            comm.Sendrecv(x[lm][1:n+1], rank+1, tag, x[0][1:n+1], rank-1, tag, status)
        # send left boundary and receive right
        if rank == 0:
            comm.Recv(x[lm+1][1:n+1], rank+1, tag, status)
        elif rank == size-1:
            comm.Send(x[1][1:n+1], rank-1, tag)
        else:
            comm.Sendrecv(x[1][1:n+1], rank-1, tag, x[lm+1][1:n+1], rank+1, tag, status)


##################################################################################################################################################################
# util.py

def write_data(lm, n, scale, psi, velfile, colfile, comm):
    # mpi essentials
    m = lm
    rank = comm.Get_rank()
    size = comm.Get_size()
    # calculate velocities and hue2rgd
    vel = np.zeros((m,n, 2))
    rgb = np.zeros((m,n,3), dtype='i')
    print(psi)
    for i in range(0, m-1):
        for j in range(0, n-1):
            vel[i][j][0] = (psi[i+1][j+2]-psi[i+1][j])/2.0
            vel[i][j][1] = -(psi[i+2][j+1]-psi[i][j+1])/2.0

            v1 = vel[i][j][0]
            v2 = vel[i][j][1]

            hue = (v1*v1 + v2*v2)**0.4    # modvsq**0.4
            rgb[i][j] = hue2rgb(hue)

    if 0 == rank:

        # Open the specified files
        velout = open(velfile, "w")
        #velout.write("{0} {1}\n".format(m/scale, n/scale))
        colout = open(colfile, "w")
        #colout.write("{0} {1}\n".format(m, n))
        for irank in range(0, size):
            if 0 == rank:
                comm.Recv(rgb[0][0][0:3*m*n], source=irank, tag=1,  status=MPI.Status())
                comm.Recv(vel[0][0][0:2*m*n], source=irank, tag=1,  status=MPI.Status())

            for irank in range(0, m):
                ix = irank*m+i+1
                for j in range(0, n):
                
                    iy = j+1
                    colout.write(f'{ix} {iy} {rgb[i][j][0]:d} {rgb[i][j][1]:d} {rgb[i][j][2]:d}\n')
                
                    #print(((ix-1)%scale, int((scale-1)/2), (iy-1)%scale, int((scale-1)/2)))
                    scale_int = int((scale-1)/2)
                    if ((ix-1)%scale == scale_int) and (iy-1)%scale == scale_int:
                        velout.write(f'{ix} {iy} {vel[i][j][0]} {vel[i][j][1]}\n')

        velout.close()
        colout.close()
    else:
        comm.Send(rgb[0][0][0:3*m*n], dest=0, tag=1)
        comm.Send(vel[0][0][0:2*m*n], dest=0, tag=1)      

def writeplotfile(m, n, scale):
    """
    Writing the plt-file to make the gnuplot
    """
    print('scalefactor', scale)
    with open('cfd.plt', 'w') as f:
        f.write('set size square\nset key off'
                   '\nunset xtics\nunset ytics\n'
                   )
        f.write(f'set xrange[{1-scale}:{m+scale}]\nset yrange[{1-scale}:{n+scale}]\n')
        f.write(f"plot \"colourmap.dat\" w rgbimage, \"velocity.dat\" u 1:2:({scale}*0.75*$3/sqrt($3**2+$4**2)):({scale}*0.75*$4/sqrt($3**2+$4**2)) with vectors  lc rgb \"#7F7F7F\"")

    print("\nWritten gnuplot script 'cfd.plt'\n");

def hue2rgb(hue):
  rgbmax = 255

  r = int(rgbmax*colfunc(hue-1.0))
  g = int(rgbmax*colfunc(hue-0.5))
  b = int(rgbmax*colfunc(hue))
  
  return int(r), int(g), int(b)


def colfunc(x):

  x1=0.2
  x2=0.5
  absx=abs(x)

  if absx > x2:
      return 0.0
  elif absx < x1:
      return 1.0
  else:
      return 1.0-((absx-x1)/(x2-x1))**2
############################################################################################################################################

# jacobi.py 

def jacobistep(psi, m, n):
    """
    Generates one step of the jacobi function for the whole grid
    """
    #return 0.25 * (psi[0:m, 1:n+1]+psi[2:m+2, 1:n+1]+psi[1:m+1,  0:n] + psi[1:m+1, 2:n+2])
    return ne.evaluate("0.25 * (a + b + c + d)", {'a':psi[0:m, 1:n+1],'b':psi[2:m+2, 1:n+1],'c':psi[1:m+1, 0:n],'d':psi[1:m+1, 2:n+2]})


def jacobistepvort(zet, psi, m, n, re):
    #print(np.sum(zet), np.sum(psi))
    #psinew = 0.25 * (psi[0:m, 1:n+1]+psi[2:m+2, 1:n+1]+psi[1:m+1, 0:n] + psi[1:m+1, 2:n+2] - zet[1:m+1, 1:n+1])
    psinew = ne.evaluate("0.25 * (a + b + c + d - e)", {'a':psi[0:m, 1:n+1],'b':psi[2:m+2, 1:n+1],'c':psi[1:m+1, 0:n],'d':psi[1:m+1, 2:n+2],'e':zet[1:m+1, 1:n+1]})

    #zetnew = - re/16.0 * ((psi[1:m+1, 2:n+2]-psi[1:m+1, 0:n])*(zet[2:m+2, 1:n+1]-zet[0:m, 1:n+1]) - (psi[2:m+2, 1:n+1]-psi[0:m, 1:n+1])*(zet[1:m+1, 2:n+2]-zet[1:m+1, 0:n])) + (0.25*(zet[0:m, 1:n+1]+zet[2:m+2, 1:n+1]+zet[1:m+1, 0:n]+zet[1:m+1, 2:n+2]))
    zetnew = ne.evaluate("- re / 16.0 * ((d - c) * (f - g) - (b - a) * (h - i)) + (0.25 * (f + g + h + i))", {'re':re,'a':psi[0:m, 1:n+1],'b':psi[2:m+2, 1:n+1],'c':psi[1:m+1, 0:n],'d':psi[1:m+1, 2:n+2],'f':zet[2:m+2, 1:n+1],'g':zet[0:m, 1:n+1],'h':zet[1:m+1, 2:n+2],'i':zet[1:m+1, 0:n]})
    return psinew, zetnew


def deltasq(psi_os_zet_temp, oldarr, m, n):
    dsq = np.sum(np.power(psi_os_zet_temp - oldarr[1: m+1, 1:n+1], 2))
    return float(dsq)

##################################################################MAIN#################################################
# cfd_numpy.py MPI4PY MAIN-file
def main(argv):
    # Test we have the correct number of arguments
    if len(argv) < 2:
        sys.stdout.write("Usage: cfd.py <scalefactor> <iterations> [reynolds]\n")
        sys.exit(1)

    # Get the systen parameters from the arguments
    scalefactor = int(argv[0])
    niter = int(argv[1])

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

    # checkreynolds
    checkerr = 0
    # //tolerance for convergence. <=0 means do not check
    tolerance = 0

  #parallelisation parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # check command line and add reynolds
    if len(argv) == 3:
        re = float(argv[2])
        irrotational = 0
        if 0 == rank:
            print(f"Reynolds number = {re}")
    else:
        re = -1
        irrotational = 1
        if 0 == rank:
            print("Irrotational flow\n")

    # irrotational?
    if not irrotational:
        zet = np.zeros((m + 2, n + 2))
    if rank == 0:
        sys.stdout.write("\n2D CFD Simulation\n")
        sys.stdout.write("=================\n")
        sys.stdout.write("Scale factor = {0}\n".format(scalefactor))
        sys.stdout.write("Iterations   = {0}\n".format(niter))
    
    # //calculate local size
    lm = int(m/size)
    
    # bnorm
    bnorm = np.array([0.0])

    # consistency check
    if size*lm != m:
        if 0 == rank:
            print(f'Error: {m} dies not divide into {size} processes')
            comm.MPI_Finalize()
    if 0 == rank:
        print(f'Running CFD on {m}x{n} grid using {size} processes')
        # Write the simulation details
        sys.stdout.write("\nGrid size = {0} x {1}\n".format(m, n))

    # didn't need it
    #print('before', scalefactor, niter, re, irrotational)
    #broadcast runtime params to other processors
    #comm.bcast(scalefactor, root=0) #  MPI_Bcast(&scalefactor,1,MPI_INT,0,comm);
    #comm.bcast(niter, root=0)   #  MPI_Bcast(&numiter,1,MPI_INT,0,comm);
    #comm.bcast(re, root=0)  #   MPI_Bcast(&re,1,MPI_DOUBLE,0,comm);
    #comm.bcast(irrotational, root=0)    #   MPI_Bcast(&irrotational,1,MPI_INT,0,comm);
    #print('after bcast', scalefactor, niter, re, irrotational)

    # reynolds number
    re = re / scalefactor

    #  //do we stop because of tolerance?
    if tolerance > 0:
        checkerr = 1


    # Define the psi array of dimension [m+2][n+2] and set it to zero
    psi = np.zeros((lm + 2, n + 2))

    # Set the psi  boundary conditions 
    boundarypsi(psi, lm, n, b, h, w, comm)

    # compute normalisation factor for error
    localbnorm = 0
    # better than double for-loop:
    localbnorm += np.sum(psi * psi)  # this is not working, just keep for the moment the iterative version

    # boundary swap of psi
    haloSWAP(psi, lm, n, comm)

    if not irrotational:
        # update zeta BCs that depends on psi
        boundaryzet(zet, psi, lm, n, comm)

        # update normalisation
        localbnorm += np.sum(zet * zet)

        # boundary swap of psi
        haloSWAP(zet, lm, n, comm)

    comm.Allreduce(sendbuf=localbnorm, recvbuf=bnorm, op=MPI.SUM)

    bnorm = np.sqrt(bnorm)

    # Call the Jacobi iterative loop (and calculate timings)
    if 0 == rank:
        sys.stdout.write("\nStarting main Jacobi loop ...\n\n")
    
    #barrier for accurate timing - not needed for correctness
    comm.Barrier()    
    
    tstart = MPI.Wtime()

    # -------------------
    for iter in range(1, niter + 1):
        # //calculate psi for next iteration
        if irrotational:
            psitmp = jacobistep(psi, lm, n)
        else:
            psitmp, zettmp = jacobistepvort(zet, psi, lm, n, re)

        # //calculate current error if required
        if checkerr or iter == niter:
            localerror = deltasq(psitmp, psi, lm, n)

            if not irrotational:
                localerror += deltasq(zettmp, zet, lm, n)

            # only rank 0 has the "error" variable!
            error = comm.reduce(localerror, op=MPI.SUM)
            if 0 == rank:
                error = np.sqrt(error) / bnorm

        # //copy back but not all!!
        psi[1:lm+1, 1:n+1] = psitmp

        if not irrotational:
            # //copy back but not all!!
            zet[1:lm+1, 1:n+1] = zettmp

        # do a boundary swap
        haloSWAP(psi, lm, n, comm)

        if not irrotational:
            haloSWAP(zet, lm, n, comm)
            # update zeta BCs that depend on psi
            boundaryzet(zet, psi, lm, n, comm)

        #       //quit early if we have reached required tolerance
        if 0 == rank and checkerr and error < tolerance:
            print(f"Converged on iteration {iter}")
            break

        # //print loop information
        if (iter % printfreq == 0) and 0 == rank:
            if not checkerr:
                print(f"Completed iteration {iter}")
            else:
                print(f"Completed iteration {iter}, error = {error}\n")

    if iter > niter:
        iter = niter
    # -------------------

    #barrier for accurate timing - not needed for correctness
    comm.Barrier()

    tend = MPI.Wtime()
    
    ttot = tend - tstart
    titer = ttot / niter
    # print out some stats
    if 0 == rank:
        print("\n... finished\n")
        print(f"After {iter}  iterations, the error is {error}\n")
        print(f"Time for {iter} iterations was {ttot} seconds\n")
        print(f"Each iteration took {titer} seconds\n")

    # Write the output files for subsequent visualisation
    #write_data(m, n, scalefactor, psi, "velocity.dat", "colourmap.dat", comm)

    # generate gnuplot file
    # Finish nicely
    if 0 == rank:
      #  writeplotfile(m, n, scalefactor)
        sys.exit(0)

    MPI.Finalize()



##############################################################
# Function to create tidy way to have main method
if __name__ == "__main__":
    main(sys.argv[1:])
##############################################################
