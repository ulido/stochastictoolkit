import numpy as np
from abc import ABC, abstractmethod
from quadtree import QuadTree
import itertools
import heapq

from .normalsrg import NormalsRG

import logging
logger = logging.getLogger(__name__)

PROCESS_VARIABLE_INITIAL_SIZE = 100

class Process(ABC, NormalsRG):
    def __init__(self, variables, time_step, boundary_condition, seed, force=None):
        NormalsRG.__init__(self, int(1e7), default_size=(1, 2), seed=seed)

        self._boundary_condition = boundary_condition
        
        self.__variables = variables
        for var, (dim, dtype) in variables.items():
            if dim == 1:
                shape = (PROCESS_VARIABLE_INITIAL_SIZE,)
            else:
                shape = (PROCESS_VARIABLE_INITIAL_SIZE, dim)
            self.__dict__['_'+var] = np.empty(shape, dtype=dtype)
        self._current_size = PROCESS_VARIABLE_INITIAL_SIZE
        self._active = np.zeros((PROCESS_VARIABLE_INITIAL_SIZE,), dtype=bool)
        self._force = np.zeros((PROCESS_VARIABLE_INITIAL_SIZE, 2), dtype=float)
        self._particle_ids = np.empty((PROCESS_VARIABLE_INITIAL_SIZE,), dtype=int)
        self._N_active = 0
        self._stale_indices = [i for i in range(self._active.shape[0])]

        self.force = force
        
        self._particle_counter = itertools.count()
        
        self.time_step = time_step
        self.time = 0

        self.__logger = logger.getChild('Process')

    def step(self): 
        if self._N_active > 0:
            new_positions = self._process_step()

            to_delete, to_reflect = self._boundary_condition(new_positions)
            if (to_reflect is not None) and to_reflect.any():
                old_positions = self._position[self._active, :][to_reflect, :]
                aidx = np.where(self._active)[0]
                to_reflect_a = aidx[to_reflect]
                not_to_reflect_a = aidx[~to_reflect]
                self._position[not_to_reflect_a, :] = new_positions[~to_reflect, :]
                crossing_points, normal_vectors = (
                    self._boundary_condition.get_crossing_and_normal(self._position[to_reflect_a, :],
                                                                      new_positions[to_reflect, :]))
                self._reflect_particles(to_reflect_a, new_positions[to_reflect, :],
                                        crossing_points, normal_vectors)
            else:
                self._position[self._active, :] = new_positions

            if to_delete is not None:
                self.remove_particles(to_delete)
        
        self.time += self.time_step

    @abstractmethod
    def _process_step(self):
        pass

    @abstractmethod
    def _reflect_particles(self, to_reflect_a, new_positions, crossing_points, tangent_vectors):
        pass

    def _pairwise_force_term(self, positions):
        if self.force is None:
            return 0
        force_obj = self.force
        forces = np.empty_like(positions)
        mi, ma = positions.min(), positions.max()
        ce = (ma+mi)/2
        hd = max((ma-mi)/1.99, 1e-5)
        qt = QuadTree([ce, ce], hd)
        qt.insert_points(positions)
        for i, q in enumerate(qt.query_self(force_obj.cutoff_distance)):
            forces[i] = force_obj(q-positions[i][np.newaxis]).sum(axis=0)

        self._force[self._active, :] = forces
        return self.time_step*forces

    def add_particle(self, **kwargs):
        self.__logger.info('Adding particle...')
        try:
            idx = heapq.heappop(self._stale_indices)
            self.__logger.debug('...with index %d' % idx)
        except IndexError:
            old_size = self._current_size
            new_size = old_size + 100
            for var, (dim, _) in self.__variables.items():
                if dim == 1:
                    self.__dict__['_' + var].resize((new_size,))
                else:
                    self.__dict__['_' + var].resize((new_size, dim))
            self._active.resize((new_size,))
            self._force.resize((new_size, 2))
            self._particle_ids.resize((new_size,))
            self._stale_indices.extend(i for i in range(old_size+1, new_size))
            self._current_size = new_size
            idx = old_size
            self.__logger.debug('...with new index %d (after extending arrays)' % idx)
        for var, value in kwargs.items():
            self.__dict__['_'+var][idx] = value
        self._active[idx] = True
        self._force[idx] = 0
        self._particle_ids[idx] = next(self._particle_counter)
        self.__logger.debug("Active index is %s" % repr(self._active))
        self._N_active += 1

    def remove_particles(self, indexes):
        if len(indexes) == 0:
            return
        self.__logger.info('Removing particles with user index %s...' % str(indexes))
        indexes = np.where(self._active)[0][indexes]
        self.__logger.debug('... which are system indexes %s' % indexes)
        for index in indexes:
            heapq.heappush(self._stale_indices, index)
        self._active[indexes] = False
        self.__logger.debug("Active index is %s" % repr(self._active))
        self._N_active -= len(indexes)

    @property
    @abstractmethod
    def parameters(self):
        return {
            'force': self.force.parameters if self.force is not None else None,
            'boundary_condition': self._boundary_condition.parameters,
        }

    @property
    def particle_ids(self):
        return self._particle_ids[self._active]

    @property
    def forces(self):
        return self._force[self._active]
