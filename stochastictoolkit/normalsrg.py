'''normalsrg.py

Contains the infrastructure to efficiently generate normal-distributed random numbers

Classes
-------
NormalsRG: Mixin class to generate normal random numbers
'''
import randomgen
import numpy as np

__all__ = ['NormalsRG']

class NormalsRG:
    '''Mixin class to get normal-distributed random numbers

    This is very basic and holds a prefilled container with normal random numbers.
    This container is refilled as needed. This is much more efficient than calling
    the RG every time we need a random number generated because it can take advantage
    of numpy optimizations and CPU vector operations.
    
    Uses the highly efficient `randomgen.Xoroshiro128` Ziggurat generator.
    '''
    def __init__(self, N_normals, default_size=(1,), seed=None):
        '''Initizialize the NormalsRG object

        Parameters
        ----------
        N_normals: int
            Number of normal random values to prefill with
        default_size: tuple
            Default array dimensions/size to return when calling `self._normal`
        seed: int
            Seed of the random number generator
        '''
        self.__N_rng = np.random.Generator(randomgen.Xoroshiro128(seed, mode='sequence'))
        
        self.__N_normals = N_normals
        self.__N_index = np.inf
        self.__N_default_size = default_size
        self.__N_default_N = np.prod(default_size)

    def __N_refill(self):
        # Refill the container and reset counter
        self.__N_array = self.__N_rng.standard_normal(size=self.__N_normals)
        self.__N_index = 0
        
    def _normal(self, size=None):
        '''Return an array of shape `size` containing normal random numbers

        If `size` is `None`, use the given `default_size`.
        '''
        if size is None:
            N = self.__N_default_N
            size = self.__N_default_size
        else:
            N = np.prod(size)

        # Refill if we run out of numbers
        if self.__N_index + N > self.__N_normals:
            self.__N_refill()
        # Get the requested numbers
        ret = self.__N_array[self.__N_index:self.__N_index+N].reshape(size)
        # Update the index
        self.__N_index += N

        return ret
    
