'''interaction_force.py

Classes
-------
InteractionForce - base class for all interaction forces
InverseDistanceForce - generic 1/r force
'''
from abc import ABC, abstractmethod

__all__ = ['InteractionForce', 'InverseDistance']

class InteractionForce:
    '''Base class for all interaction forces

    All interaction forces inherit from this class. When subclassing the following
    abstract methods need to be implemented:
    * `__call__`: calculates the force for the given distances and return it
    * `parameters`: property returning the force parameters

    '''
    def __init__(self, cutoff_distance):
        '''Initialize InteractionForce

        Parameters
        ----------
        cutoff_distance: float
            Cutoff distance for the neighborhood search (typically chosen s.t. the
            force has become negligible)
        '''
        self.cutoff_distance = cutoff_distance

    @abstractmethod
    def __call__(self, x):
        '''Calculate force from the distance and return'''
        pass

    @property
    @abstractmethod
    def parameters(self):
        '''InteractionForce parameters

        A subclass needs to override this function and retrieve the Process parameters from `Process.parameters`.
        '''
        return {
            'cutoff_distance': self.cutoff_distance
        }

class InverseDistanceForce(InteractionForce):
    '''1/r potential force class

    This calculates the force
    \[\vec{F}(\vec{x})=f_0\frac{\vec{x}}{|\vec{x}|^3}\]
    '''
    def __init__(self, strength, cutoff_distance):
        '''Initialize InverseDistanceForce

        Parameters
        ----------
        strength: float
            Force strength $f_0$
        cutoff_distance: float
            Neighborhood search cutoff distance
        '''
        super().__init__(cutoff_distance=cutoff_distance)
        self.strength = strength

    def __call__(self, x):
        '''Calculate force from distances and return.'''
        return -self.strength*x/(x**2).sum(axis=1)**(1.5)

    @property
    def parameters(self):
        '''Force parameters'''
        ret = super().parameters
        ret.update({
            'name': 'OneOverRForce',
            'strength': self.strength
        })
        return ret
