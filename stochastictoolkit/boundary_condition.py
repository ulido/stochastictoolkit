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

    All boundary condition classes inherit from this class.

    When subclassing the following methods can be implemented:
    * `absorbing_boundary`: decides if a particle has crossed an absorbing boundary
    * `reflecting_boundary`: decides if a particle has crossed a reflecting boundary
    * `periodic_boundary`: decides if a particle has crossed a periodic boundary
    * `get_reflective_crossing_and_normal`: calculates crossing point and normal vector for reflecting
      bounds (mandatory if `reflecting_boundary` is implemented and `true_reflection` evaluates to `True`)
    * `get_periodic_new_position`: calculates the new position when crossing a periodic boundary
      (mandatory if `periodic_boundary` is implemented)
    * `true_reflection` (property): decides if to use "true" reflection or simply disallow updates
    * `periodic_ghost_positions`: checks if position is within `offset` of a periodic
      boundary and returns the corresponding ghost positions (mandatory if `periodic_boundary`
      is implemented and particle interaction forces are in use)
    * `__str__`: boundary object description (mandatory)
    * `parameters`: property returning the boundary parameters
    '''
    def absorbing_boundary(self, positions):
        '''Evaluate absorbing boundaries

        For each particle position in `positions` evaluate if the particle has crossed
        an absorbing boundary. Return a bool array - true for particles that are absorbed,
        false else. Return `None` if no absorbing boundaries exist.
        '''
        return None
        
    def reflecting_boundary(self, positions):
        '''Evaluate reflecting boundaries

        For each particle position in `positions` evaluate if the particle has crossed
        a reflecting boundary. Return a bool array - true for particles that need to be reflected,
        false else. Return `None` if no reflecting boundaries exist.
        '''
        return None

    def periodic_boundary(self, positions):
        '''Evaluate periodic boundaries

        For each particle position in `positions` evaluate if the
        particle has crossed a reflecting boundary. Return a bool
        array (True if particle position needs to be updated, False if
        not).
        '''
        return None

    def get_reflective_crossing_and_normal(self, positions, new_positions):
        '''Return the crossing point and the normal vector for each particle'''
        raise NotImplementedError

    def get_periodic_new_position(self, positions):
        '''Return the new positions when crossing a periodic boundary'''
        raise NotImplementedError

    def periodic_ghost_positions(self, positions, offset):
        '''Return a bool array of positions that are within offset of a periodic boundary.'''
        return None

    @property
    def true_reflection(self):
        '''Whether we want true reflection of particles or simple disallowed updates.

        We set true reflection as the default, but a subclass can override this in case the
        accuracy is not worth the computational overhead. 
        '''
        return True
    
    @property
    @abstractmethod
    def parameters(self):
        '''BoundaryCondition parameters'''
        return {
            'true reflection': self.true_reflection,
        }

class NoBoundaries(BoundaryCondition):
    '''No boundaries boundary condition'''

    @property
    def parameters(self):
        '''Parameters of the boundary condition'''
        ret = super().parameters
        ret.update({
            'name': 'NoBoundaries',
        })
        return ret
