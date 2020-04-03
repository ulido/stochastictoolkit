import randomgen
import numpy as np

import logging
logger = logging.getLogger(__name__)

class NormalsRG:
    def __init__(self, N_normals, default_size=(1,), seed=None):
        self.__N_rng = randomgen.Generator(randomgen.Xoroshiro128(seed, mode='sequence'))
        
        self.__N_normals = N_normals
        self.__N_index = np.inf
        self.__N_default_size = default_size
        self.__N_default_N = np.prod(default_size)

        self.__logger = logger.getChild('NormalsRG')
        
    def __N_refill(self):
        self.__logger.info('Refilling random numbers')
        self.__N_array = self.__N_rng.standard_normal(size=self.__N_normals)
        self.__N_index = 0
        
    def _normal(self, size=None):
        if size is None:
            N = self.__N_default_N
            size = self.__N_default_size
        else:
            N = np.prod(size)
        if self.__N_index + N > self.__N_normals:
            self.__N_refill()
        ret = self.__N_array[self.__N_index:self.__N_index+N].reshape(size)
        self.__N_index += N
        return ret
    
