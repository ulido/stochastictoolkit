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
    def __init__(self, variables, time_step, boundary_condition, seed,
                 force_function=None, force_strength=0,
                 force_cutoff_distance=0):
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
        self._particle_ids = np.empty((PROCESS_VARIABLE_INITIAL_SIZE,), dtype=int)
        self._N_active = 0
        self._stale_indices = [i for i in range(self._active.shape[0])]

        if (force_strength != 0) and (force_function is None):
            raise ValueError("Force strength is nonzero but no force function specified!")
        if (force_strength != 0) and (force_cutoff_distance <= 0):
            raise ValueError("Force strength is nonzero but cutoff distance is invalid!")
        self._force_cutoff_distance = force_cutoff_distance
        self._force_strength_dt = force_strength*time_step
        self._force_function = force_function
        
        self._particle_counter = itertools.count()
        
        self.time_step = time_step
        self.time = 0

        self.__logger = logger.getChild('Process')

    def step(self): 
        if self._N_active > 0:
            new_positions = self._process_step()

            to_delete, to_update = self._boundary_condition(new_positions)
            to_update_a = np.where(self._active)[0][to_update]
            self._position[to_update_a, :] = new_positions[to_update, :]
            self.remove_particles(to_delete)
        
        self.time += self.time_step

    @abstractmethod
    def _process_step(self):
        pass

    def _pairwise_force_term(self, positions):
        if self._force_strength_dt == 0:
            return 0
        mi, ma = positions.min(), positions.max()
        ce = (ma+mi)/2
        hd = max((ma-mi)/1.99, 1e-5)
        qt = QuadTree([ce, ce], hd)
        qt.insert_points(positions)
        forces = np.empty_like(positions)
        for i, q in enumerate(qt.query_self(self._force_cutoff_distance)):
            forces[i] = self._force_function(q-positions[i][np.newaxis]).sum(axis=0)
        return self._force_strength_dt*forces

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
            self._particle_ids.resize((new_size,))
            self._stale_indices.extend(i for i in range(old_size+1, new_size))
            self._current_size = new_size
            idx = old_size
            self.__logger.debug('...with new index %d (after extending arrays)' % idx)
        for var, value in kwargs.items():
            self.__dict__['_'+var][idx] = value
        self._active[idx] = True
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
            'force_strength': self._force_strength_dt/self.time_step,
            'force_function': str(self._force_function),
            'force_cutoff_distance': self._force_cutoff_distance,
        }

    @property
    def particle_ids(self):
        return self._particle_ids[self._active]
