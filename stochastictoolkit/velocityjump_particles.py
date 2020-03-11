import numpy as np
import heapq

from stochastictoolkit.brownian_particles import NormalsRG

import logging

logger = logging.getLogger(__name__)

class VelocityJumpProcess(NormalsRG):
    def __init__(self, time_step, diffusion_coefficient, position_diffusion_coefficient=1, model='Gradient',
                boundary_condition=None, seed=None, recorder=None, field=None, gamma=0., speed=1):
        NormalsRG.__init__(self, int(1e7), default_size=(1, 2), seed=seed)
        self.__time_step = time_step
        self.__stepsize = (2*diffusion_coefficient*time_step)**0.5
        self.__pos_stepsize = (2*position_diffusion_coefficient*time_step)**0.5
        self.__gammadt = gamma*time_step
        self.__speeddt = speed*time_step
        self.__boundary_condition = boundary_condition

        self.__positions = np.empty((100, 2), dtype=float)
        self.__angles = np.empty((100,), dtype=float)
        self.__active = np.zeros((self.__positions.shape[0],), dtype=bool)
        self.__N_active = 0
        self.__stale_indices = [i for i in range(self.__positions.shape[0])]
        self.time = 0
        
        self.__model = model
        
        self.field = field
        
        self.__logger = logger.getChild('VelocityJump')
        
    #@profile
    def step(self):
        #self.__logger.debug('Performing EM step')
        if self.__N_active > 0:
            if self.__model == 'Perna':
                velocities = np.exp(1j*self.__angles[self.__active]).view(np.float).reshape((-1, 2))
                new_pos = self.__positions[self.__active, :] + 0.1*self.__time_step*velocities

                ps = self.__positions[self.__active]
                angles = self.__angles[self.__active]
                new_angles = np.empty_like(angles)
                for i in range(angles.shape[0]):
                    d = ps[i][np.newaxis]-self.field._points
                    dist = np.linalg.norm(d, axis=1)
                    inrange = dist < 0.05

                    points = d[inrange]
                    pangle = np.arctan2(points[:, 1], points[:, 0]) - angles[i]
                    pangle[pangle < -np.pi] += 2*np.pi
                    pangle[pangle > np.pi] -= 2*np.pi
                    left = (pangle > -np.pi/2) & (pangle < 0)
                    right = (pangle < np.pi/2) & (pangle >= 0)
                    values = self.field._lattice[inrange.reshape(self.field._lattice.shape)]

                    L = values[left].mean()
                    R = values[right].mean()

                    new_angles[i] = angles[i] + 10*(R-L)
                # Implement wraparound!
                self.__angles[self.__active] = new_angles
                
                to_delete, to_update = self.__boundary_condition(new_pos)
                to_update_a = np.where(self.__active)[0][to_update]
                self.__positions[to_update_a, :] = new_pos[to_update, :]
                self.remove_particles(to_delete)
            elif self.__model == 'Gradient':
                positions = self.__positions[self.__active]
                angles = self.__angles[self.__active]
                
                velocities = np.exp(1j*(angles)).view(np.float).reshape(-1, 2)
                normals = np.vstack((-velocities[:, 1], velocities[:, 0])).T

                grad_c = self.field._lattice.get_gradient(positions)
                drift = self.__gammadt*(normals*grad_c).sum(axis=1)
                diffusion = self.__stepsize*self._normal(size=(self.__N_active,))
                self.__angles[self.__active] += drift + diffusion
                
                new_pos = positions + velocities*self.__speeddt + self.__pos_stepsize*self._normal(size=(self.__N_active, 2))

                to_delete, to_update = self.__boundary_condition(new_pos)
                to_update_a = np.where(self.__active)[0][to_update]
                self.__positions[to_update_a, :] = new_pos[to_update, :]
                self.remove_particles(to_delete)
            else:
                raise ValueError(f'Unknown turning model {self.__model}')
                
        self.time += self.__time_step
        
    def add_particle(self, position):
        #self.__logger.info('Adding particle...')
        try:
            idx = heapq.heappop(self.__stale_indices)
            #self.__logger.debug('...with index %d' % idx)
        except IndexError:
            old_size = self.__positions.shape[0]
            # Need to set refcheck=False when profiling?
            self.__positions.resize((old_size+100, 2), refcheck=False)
            self.__angles.resize((old_size+100,), refcheck=False)
            self.__active.resize((old_size+100,), refcheck=False)
            self.__stale_indices.extend(i for i in range(old_size+1, old_size+100))
            idx = old_size
            #self.__logger.debug('...with new index %d (after extending arrays)' % idx)
        self.__positions[idx] = position
        self.__angles[idx] = 2*np.pi*np.random.uniform()
        self.__active[idx] = True
        #self.__logger.debug("Active index is %s" % repr(self.__active))
        self.__N_active += 1
        
    def remove_particles(self, indexes):
        if len(indexes) == 0:
            return
        self.__logger.info('Removing particles with user index %s...' % str(indexes))
        indexes = np.where(self.__active)[0][indexes]
        self.__logger.debug('... which are system indexes %s' % indexes)
        for index in indexes:
            heapq.heappush(self.__stale_indices, index)
        self.__active[indexes] = False
        self.__logger.debug("Active index is %s" % repr(self.__active))
        self.__N_active -= len(indexes)

    @property
    def positions(self):
        return self.__positions[self.__active, :]
    
    @property
    def velocities(self):
        return self.__speeddt/self.__time_step*np.exp(1j*self.__angles[self.__active]).view(np.float).reshape(-1, 2)

