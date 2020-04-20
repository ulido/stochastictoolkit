'''boundary_condition.py

Classes
-------
BoundaryCondition - base class for all boundary conditions
NoBoundaries - for domains without any boundaries
'''
from abc import ABC, abstractmethod

__all__ = ['BoundaryCondition', 'NoBoundaries']

class BoundaryCondition(ABC):
    '''Base class for all boundary conditions

    All boundary condition classes inherit from this class When subclassing the following
    abstract methods need to be implemented:
    * `absorbing_boundary`: decides if a particle has crossed an absorbing boundary
    * `reflecting_boundary`: decides if a particle has crossed a reflecting boundary
    * `get_crossing_and_normal`: calculates crossing point and normal vector for reflecting bounds
    * `__str__`: boundary object description
    * `parameters`: property returning the boundary parameters
    '''
    @abstractmethod
    def absorbing_boundary(self, positions):
        '''Evaluate absorbing boundaries

        For each particle position in `positions` evaluate if the particle has crossed
        an absorbing boundary. Return a bool array - true for particles that are absorbed,
        false else. Return `None` if no absorbing boundaries exist.
        '''
        pass
        
    @abstractmethod
    def reflecting_boundary(self, positions):
        '''Evaluate reflecting boundaries

        For each particle position in `positions` evaluate if the particle has crossed
        a reflecting boundary. Return a bool array - true for particles that need to be reflected,
        false else. Return `None` if no reflecting boundaries exist.
        '''
        pass

    @abstractmethod
    def get_crossing_and_normal(self, positions, new_positions):
        '''Return the crossing point and the normal vector for each particle'''
        pass

    @abstractmethod
    def __str__(self):
        return "Boundary Condition base class (abstract)"

    def __call__(self, positions):
        '''Evaluate boundary condition'''
        to_delete = self.absorbing_boundary(positions)
        to_reflect = self.reflecting_boundary(positions)
        return to_delete, to_reflect

    @property
    @abstractmethod
    def parameters(self):
        '''BoundaryCondition parameters'''
        return {}

class NoBoundaries(BoundaryCondition):
    '''No boundaries boundary condition'''
    def absorbing_boundary(self, positions):
        return None
    
    def reflecting_boundary(self, positions):
        return None
    
    def get_crossing_and_normal(self, positions, new_positions):
        # There are no reflective boundaries... and this can't be called anyway.
        raise NotImplementedError()

    def __str__(self):
        return "No boundary condition"

    @property
    def parameters(self):
        '''Parameters of the boundary condition'''
        ret = super().parameters
        ret.update({
            'name': 'NoBoundaries',
        })
        return ret
