import numpy as np

from .particle_type import Process

class BrownianProcess(Process):
    def __init__(self,
                 time_step,
                 diffusion_coefficient,
                 boundary_condition,
                 seed=None):
        variables = {
            'position': (2, float)
        }
        super().__init__(variables, time_step, seed)

        self.__diffusion_coefficient = diffusion_coefficient

        self.__stepsize = (2*diffusion_coefficient*time_step)**0.5
        self.__boundary_condition = boundary_condition

    @property
    def parameters(self):
        return {
            'process': 'BrownianProcess',
            'time_step': self.time_step,
            'diffusion_coefficient': self.__diffusion_coefficient,
        }
    
    def _process_step(self):
        if self._N_active > 0:
            new_pos = self._position[self._active, :] + self.__stepsize*self._normal(size=(self._N_active, 2))
            to_delete, to_update = self.__boundary_condition(new_pos)
            to_update_a = np.where(self._active)[0][to_update]
            self._position[to_update_a, :] = new_pos[to_update, :]
            self.remove_particles(to_delete)
        
    @property
    def positions(self):
        return self._position[self._active, :]
                
