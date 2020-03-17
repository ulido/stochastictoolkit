import numpy as np
from ._PDE import get_gradient, get_gradient_9pstencil
from tqdm.auto import tqdm

class CoordinateLattice(np.ndarray):
    def __new__(subtype, lattice_size, dtype=float, buffer=None, offset=0, strides=None, order=None, dx=1, origin=None):
        shape = (lattice_size/dx).astype(int)
        obj = super(CoordinateLattice, subtype).__new__(subtype, shape, dtype,
                                                        buffer, offset, strides,
                                                        order)
        
        obj.dx = dx
        if origin is None:
            obj.origin = np.array(shape, dtype=int)//2
        else:
            obj.origin = np.array(origin, dtype=int)
                    
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None:
            return
        
        self.dx = getattr(obj, 'dx', 1)
        self.origin = getattr(obj, 'origin', np.array([0, 0], dtype=int))
        
    def __indexes_from_positions(self, points):
        points = np.atleast_2d(points)
        return np.around(points/self.dx).astype(int) + self.origin[np.newaxis]
        
    def add_values(self, points, values):
        idxs = self.__indexes_from_positions(points)
        self[idxs[:, 0], idxs[:, 1]] += values
        
    def get_values(self, points):
        idxs = self.__indexes_from_positions(points)
        return self[idxs[:, 0], idxs[:, 1]]
    
    def get_gradient(self, points):
        points = np.atleast_2d(points)
        ret = np.empty((points.shape[0], 2))
        get_gradient_9pstencil(self, points, ret)
        return ret

class DiffusionPDESolver:
    def __init__(self,
                 lattice_size,
                 diffusion_coefficient,
                 source_strength,
                 source_positions,
                 decay_rate,
                 dx,
                 max_dt=None):
        lattice_size = np.array(lattice_size)
        self._lattice = CoordinateLattice(lattice_size, dx=dx)
        self._lattice[:] = 0.
        self._source_positions = source_positions
        
        self._source_strength = source_strength
        self._source_positions = source_positions
        self._diffusion_coefficient = diffusion_coefficient
        self._dx = dx
        self._decay_rate = decay_rate

        self.time = 0
        
        try:
            self._dt = 0.5/(4*diffusion_coefficient/dx**2+decay_rate)
            if max_dt is not None:
                self._dt = min(self._dt, max_dt)
        except ZeroDivisionError:
            if max_dt is not None:
                self._dt = max_dt
            else:
                raise ValueError('Cannot calculate a time stap and max_dt was not specified.')

        self._r = self._diffusion_coefficient*self._dt/self._dx**2
        self._m = (1 - 4*self._r - self._decay_rate*self._dt)
        self._source_strength_dt = source_strength*self._dt
    
    def step(self):
        self.time += self._dt

        n = np.zeros_like(self._lattice)
        c = self._lattice        
        n[1:-1, 1:-1] = (
            self._m * c[1:-1, 1:-1] +
            self._r * (c[2:, 1:-1] + c[:-2, 1:-1] + 
                 c[1:-1, 2:] + c[1:-1, :-2])
        )
        if self._source_positions is not None:
            n.add_values(self._source_positions(), self._source_strength_dt)

        self._lattice = n

    @property
    def parameters(self):
        return {
            'diffusion_coefficient': self._diffusion_coefficient,
            'time_step': self._dt,
            'dx': self._dx,
            'source_strength': self._source_strength,
            'source_positions': str(self._source_positions),
            'decay_rate': self._decay_rate,
        }
