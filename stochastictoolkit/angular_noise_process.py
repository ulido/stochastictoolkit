import numpy as np
import heapq

from stochastictoolkit.particle_type import Process

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
                 seed=None):
        variables = {
            'position': (2, float),
            'angle': (1, float),
        }
        super().__init__(variables, time_step, seed)

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
        self.__boundary_condition = boundary_condition

        self.time = 0
        
        self.__logger = logger.getChild('VelocityJump')

    @property
    def parameters(self):
        return {
            'process': 'AngularNoiseProcessWithAngularDrift',
            'time_step': self.__time_step,
            'position_diffusion_coefficient': self.__position_diffusion_coefficient,
            'angular_diffusion_coefficient': self.__angular_diffusion_coefficient,
            'drift_strength': self.__drift_strength,
            'drift_function': str(self.__drift_function),
            'speed': self.__speed,
        }
        
    def _process_step(self):
        #self.__logger.debug('Performing EM step')
        if self._N_active > 0:
            positions = self._position[self._active]
            angles = self._angle[self._active]
            
            velocities = np.exp(1j*(angles)).view(np.float).reshape(-1, 2)

            if self.__drift_function is not None:
                drift = self.__drift_strength_dt*self.__drift_function(positions, velocities, self.time)
            else:
                drift = 0.
            #grad_c = self.field._lattice.get_gradient(positions)
            #drift = self.__gammadt*(normals*grad_c).sum(axis=1)
            diffusion = self.__angular_stepsize*self._normal(size=(self._N_active,))
            self._angle[self._active] += drift + diffusion
            
            new_pos = positions + velocities*self.__speeddt + self.__pos_stepsize*self._normal(size=(self._N_active, 2))
            to_delete, to_update = self.__boundary_condition(new_pos)
            to_update_a = np.where(self._active)[0][to_update]
            self._position[to_update_a, :] = new_pos[to_update, :]
            self.remove_particles(to_delete)

            # if self.__model == 'Perna':
            #     velocities = np.exp(1j*self.__angles[self.__active]).view(np.float).reshape((-1, 2))
            #     new_pos = self.__positions[self.__active, :] + 0.1*self.__time_step*velocities

            #     ps = self.__positions[self.__active]
            #     angles = self.__angles[self.__active]
            #     new_angles = np.empty_like(angles)
            #     for i in range(angles.shape[0]):
            #         d = ps[i][np.newaxis]-self.field._points
            #         dist = np.linalg.norm(d, axis=1)
            #         inrange = dist < 0.05

            #         points = d[inrange]
            #         pangle = np.arctan2(points[:, 1], points[:, 0]) - angles[i]
            #         pangle[pangle < -np.pi] += 2*np.pi
            #         pangle[pangle > np.pi] -= 2*np.pi
            #         left = (pangle > -np.pi/2) & (pangle < 0)
            #         right = (pangle < np.pi/2) & (pangle >= 0)
            #         values = self.field._lattice[inrange.reshape(self.field._lattice.shape)]

            #         L = values[left].mean()
            #         R = values[right].mean()

            #         new_angles[i] = angles[i] + 10*(R-L)
            #     # Implement wraparound!
            #     self.__angles[self.__active] = new_angles
                
            #     to_delete, to_update = self.__boundary_condition(new_pos)
            #     to_update_a = np.where(self.__active)[0][to_update]
            #     self.__positions[to_update_a, :] = new_pos[to_update, :]
            #     self.remove_particles(to_delete)
        
    def add_particle(self, position):
        super().add_particle(position=position, angle=2*np.pi*np.random.uniform())
        
    @property
    def positions(self):
        return self._position[self._active, :]
    
    @property
    def velocities(self):
        return self.__speeddt/self.__time_step*np.exp(1j*self._angle[self._active]).view(np.float).reshape(-1, 2)
