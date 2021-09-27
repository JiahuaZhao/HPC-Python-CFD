#
# Create files of input data (for plotting) intended for the
# accompanying "plot_flow.py" script. 
#
# ARCHER, 2015
#
# Note that the colour map is written at 'full' resolution,
# while the velocity field is written at a resolution which
# depends on the scale factor.

import sys
import numpy as np

def write_data(m, n, scale, psi, velfile, colfile):

    # calculate velocities and hue2rgd
    vel = np.zeros((m,n, 2))
    rgb = np.zeros((m,n,3), dtype='i')
    for i in range(0, m):
        for j in range(0, n):
	        vel[i][j][0] =  (psi[i+1][j+2]-psi[i+1][j])/2.0
	        vel[i][j][1] = -(psi[i+2][j+1]-psi[i][j+1])/2.0

	        v1 = vel[i][j][0]
	        v2 =  vel[i][j][1]

	        hue = (v1*v1 + v2*v2)**0.4    # modvsq**0.4

	        rgb[i][j] = hue2rgb(hue)

    # Open the specified files
    velout = open(velfile, "w")
    #velout.write("{0} {1}\n".format(m/scale, n/scale))
    colout = open(colfile, "w")
    #colout.write("{0} {1}\n".format(m, n))

    for i in range(0, m):
        ix = i+1
        for j in range(0, n):
	    
            iy = j+1
            colout.write(f'{ix} {iy} {rgb[i][j][0]:d} {rgb[i][j][1]:d} {rgb[i][j][2]:d}\n')
	    
            #print(((ix-1)%scale, int((scale-1)/2), (iy-1)%scale, int((scale-1)/2)))
            scale_int = int((scale-1)/2)
            if ((ix-1)%scale == scale_int) and (iy-1)%scale == scale_int:
                velout.write(f'{ix} {iy} {vel[i][j][0]} {vel[i][j][1]}\n')

    """
    # Loop over stream function array (excluding boundaries)
    for i in range(1, m+1):
        for j in range(1, n+1):

            # Compute velocities and magnitude
            ux =  (psi[i][j+1] - psi[i][j-1])/2.0
            uy = -(psi[i+1][j] - psi[i-1][j])/2.0
            umod = (ux**2 + uy**2)**0.5

            # We are actually going to output a colour, in which
            # case it is useful to shift values towards a lighter
            # blue (for clarity) via the following kludge...

            hue = umod**0.6
            colout.write("{0:5d} {1:5d} {2:10.5f}\n".format(i-1, j-1, hue))

            # Only write velocity vectors every "scale" points
            if (i-1)%scale == (scale-1)/2 and (j-1)%scale == (scale-1)/2:
                velout.write("{0:5d} {1:5d} {2:10.5f} {3:10.5f}\n".format(i-1, j-1, ux, uy))
    """
    velout.close()
    colout.close()

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
