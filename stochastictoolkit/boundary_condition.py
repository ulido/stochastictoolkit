from abc import ABC, abstractmethod

class BoundaryCondition(ABC):
    @abstractmethod
    def absorbing_boundary(self, positions):
        # Return a bool array - true for particles that are absorbed, false else
        pass
        
    @abstractmethod
    def reflecting_boundary(self, positions):
        # Return a bool array - true for particles that need to be reflected, false else
        pass

    @abstractmethod
    def get_crossing_and_normal(self, positions, new_positions):
        # Return the crossing point and the normal vector
        pass

    @abstractmethod
    def __str__(self):
        return "Boundary Condition base class (abstract)"

    def __call__(self, positions):
        to_delete = self.absorbing_boundary(positions)
        to_reflect = self.reflecting_boundary(positions)
        return to_delete, to_reflect

    @property
    @abstractmethod
    def parameters(self):
        return {}

class NoBoundaries(BoundaryCondition):
    def absorbing_boundary(self, positions):
        return None
    
    def reflecting_boundary(self, positions):
        return None
    
    def get_crossing_and_normal(self, positions, new_positions):
        # There are no reflective boundaries...
        raise NotImplementedError()

    def __str__(self):
        return "No boundary condition"

    @property
    def parameters(self):
        ret = super().parameters
        ret.update({
            'name': 'NoBoundaries',
        })
        return ret
