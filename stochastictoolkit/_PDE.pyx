# cython: language_level=3

from libc.math cimport lround
import numpy as np
cimport cython

@cython.cdivision(True)
@cython.boundscheck(False)
def get_gradient(lattice, const double[:, :] positions, double[:, :] out):
    cdef int i, j, k
    cdef double v
    cdef double[:, :] c = lattice
    cdef double dx = lattice.dx
    cdef int xorigin = lattice.origin[0]
    cdef int yorigin = lattice.origin[1]
    
    for k in range(positions.shape[0]):
        i = lround(positions[k, 0]/dx) + xorigin
        j = lround(positions[k, 1]/dx) + yorigin
        
        out[k, 0] = (c[i+1, j] - c[i-1, j])/2
        out[k, 1] = (c[i, j+1] - c[i, j-1])/2


@cython.cdivision(True)
@cython.boundscheck(False)
def get_gradient_9pstencil(lattice, const double[:, :] positions, double[:, :] out):
    cdef int i, j, k
    cdef double v
    cdef double[:, :] c = lattice
    cdef double dx = lattice.dx
    cdef int xorigin = lattice.origin[0]
    cdef int yorigin = lattice.origin[1]
    
    for k in range(positions.shape[0]):
        i = lround(positions[k, 0]/dx) + xorigin
        j = lround(positions[k, 1]/dx) + yorigin

        out[k, 0] = ((c[i+1, j+1] + 4*c[i+1, j] + c[i+1, j-1]) -
                     (c[i-1, j+1] + 4*c[i-1, j] + c[i-1, j-1]))/12.0 
        out[k, 1] = ((c[i+1, j+1] + 4*c[i, j+1] + c[i-1, j+1]) -
                     (c[i+1, j-1] + 4*c[i, j-1] + c[i-1, j-1]))/12.0


