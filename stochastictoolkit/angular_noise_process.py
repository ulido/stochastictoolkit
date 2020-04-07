import numpy as np
import heapq

from .process import Process

import logging

logger = logging.getLogger(__name__)

class AngularNoiseProcessWithAngularDrift(Process):
    def __init__(self,
                 time_step,
                 boundary_condition,
                 angular_diffusion_coefficient,
                 position_diffusion_coefficient,
                 drift_strength,
                 speed,
                 drift_function=None,
                 force_strength=0.,
                 force_function=None,
                 force_cutoff_distance=0,
                 seed=None):
        variables = {
            'position': (2, float),
            'angle': (1, float),
        }
        super().__init__(variables, time_step, boundary_condition, seed,
                         force_strength=force_strength,
                         force_function=force_function,
                         force_cutoff_distance=force_cutoff_distance)

        self.__time_step = time_step
        self.__angular_diffusion_coefficient = angular_diffusion_coefficient
        self.__position_diffusion_coefficient = position_diffusion_coefficient
        self.__drift_strength = drift_strength
        self.__speed = speed
        
        self.__angular_stepsize = (2*angular_diffusion_coefficient*time_step)**0.5
        self.__pos_stepsize = (2*position_diffusion_coefficient*time_step)**0.5
        self.__drift_strength_dt = drift_strength*time_step
        self.__drift_function = drift_function
        self.__speeddt = speed*time_step

        self.time = 0
        
        self.__logger = logger.getChild('VelocityJump')

    @property
    def parameters(self):
        ret = super().parameters
        ret.update({
            'process': 'AngularNoiseProcessWithAngularDrift',
            'time_step': self.__time_step,
            'position_diffusion_coefficient': self.__position_diffusion_coefficient,
            'angular_diffusion_coefficient': self.__angular_diffusion_coefficient,
            'drift_strength': self.__drift_strength,
            'drift_function': str(self.__drift_function),
            'speed': self.__speed,
        })
        return ret
        
    def _process_step(self):
        positions = self._position[self._active]
        angles = self._angle[self._active]
        
        velocities = np.exp(1j*(angles)).view(np.float).reshape(-1, 2)
        
        if self.__drift_function is not None:
            drift = self.__drift_strength_dt*self.__drift_function(positions, velocities, self.time)
        else:
            drift = 0.
        diffusion = self.__angular_stepsize*self._normal(size=(self._N_active,))
        self._angle[self._active] += drift + diffusion
        
        pos_drift = self._pairwise_force_term(positions) + velocities*self.__speeddt
        pos_diffusion = self.__pos_stepsize*self._normal(size=(self._N_active, 2))
        return positions + pos_drift + pos_diffusion

    def add_particle(self, position, angle=None):
        if angle is None:
            angle = 2*np.pi*np.random.uniform()
        super().add_particle(position=position, angle=angle)
        
    @property
    def positions(self):
        return self._position[self._active, :]
    
    @property
    def velocities(self):
        return self.__speeddt/self.__time_step*np.exp(1j*self._angle[self._active]).view(np.float).reshape(-1, 2)
