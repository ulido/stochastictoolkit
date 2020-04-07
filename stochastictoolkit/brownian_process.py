import numpy as np

from .process import Process

class BrownianProcess(Process):
    def __init__(self,
                 time_step,
                 diffusion_coefficient,
                 boundary_condition,
                 force_strength=0.,
                 force_function=None,
                 force_cutoff_distance=0,
                 seed=None):
        variables = {
            'position': (2, float)
        }
        super().__init__(variables, time_step, boundary_condition, seed,
                         force_strength=force_strength,
                         force_function=force_function,
                         force_cutoff_distance=force_cutoff_distance)

        self.__diffusion_coefficient = diffusion_coefficient

        self.__stepsize = (2*diffusion_coefficient*time_step)**0.5

    @property
    def parameters(self):
        ret = super().parameters
        ret.update({
            'process': 'BrownianProcess',
            'time_step': self.time_step,
            'diffusion_coefficient': self.__diffusion_coefficient,
        })
        return ret
    
    def _process_step(self):
        positions = self._position[self._active, :]
        drift = self._pairwise_force_term(positions)
        diffusion = self.__stepsize*self._normal(size=(self._N_active, 2))
        return positions + drift + diffusion

    def _reflect_particles(self, to_reflect_a, new_positions, crossing_points, tangent_vectors):
        d = new_positions - crossing_points
        dotp = (d*tangent_vectors).sum(axis=1)
        self._position[to_reflect_a] = crossing_points-d+2*dotp[:, np.newaxis]*tangent_vectors

    @property
    def positions(self):
        return self._position[self._active, :]
                
