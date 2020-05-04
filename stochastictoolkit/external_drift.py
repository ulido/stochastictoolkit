'''external_drift.py

Classes
-------
ExternalDrift - base class for all external drift objects
ConstantDrift - generic class to apply a constant drift
'''
from abc import ABC, abstractmethod
import numpy as np

__all__ = ['ExternalDrift', 'ConstantDrift']

class ExternalDrift:
    '''Base class for all external drift classes

    All external drift classes inherit from this class. When subclassing the following
    abstract methods need to be implemented:
    * `__call__`: calculates the external drift at the given positions
    * `parameters`: property returning the drift parameters

    '''
    def __init__(self):
        '''Initialize ExternalDrift
        '''
        pass

    @abstractmethod
    def __call__(self, x):
        '''Calculate drift from the position and return'''
        pass

    @property
    @abstractmethod
    def parameters(self):
        '''ExternalDrift parameters

        A subclass needs to override this function and retrieve the ExternalDrift parameters from `ExternalDrift.parameters`.
        '''
        return {}

class ConstantDrift(ExternalDrift):
    '''Constant external drift class
    '''
    def __init__(self, drift_vector):
        '''Initialize ConstantDrift

        Parameters
        ----------
        drift_vector: ndarray of size Ndim
            Constant drift vector
        '''
        super().__init__()
        self.drift_vector = np.array(drift_vector)[np.newaxis]

    def __call__(self, x):
        '''Calculate drift at the given positions.'''
        return self.drift_vector

    @property
    def parameters(self):
        '''ConstantDrift parameters'''
        ret = super().parameters
        ret.update({
            'name': 'ConstantDrift',
            'drift_vector': self.drift_vector[0],
        })
        return ret
