import numpy as np
import heapq
import randomgen
import itertools
from abc import ABC, abstractmethod
from quadtree import QuadTree
import logging

logger = logging.getLogger(__name__)

class NormalsRG:
    def __init__(self, N_normals, default_size=(1,), seed=None):
        self.__N_rng = randomgen.Generator(randomgen.Xoroshiro128(seed, mode='sequence'))
        
        self.__N_normals = N_normals
        self.__N_index = np.inf
        self.__N_default_size = default_size
        self.__N_default_N = np.prod(default_size)

        self.__logger = logger.getChild('NormalsRG')
        
    def __N_refill(self):
        self.__logger.info('Refilling random numbers')
        self.__N_array = self.__N_rng.standard_normal(size=self.__N_normals)
        self.__N_index = 0
        
    def _normal(self, size=None):
        if size is None:
            N = self.__N_default_N
            size = self.__N_default_size
        else:
            N = np.prod(size)
        if self.__N_index + N > self.__N_normals:
            self.__N_refill()
        ret = self.__N_array[self.__N_index:self.__N_index+N].reshape(size)
        self.__N_index += N
        return ret
    
class BoundaryCondition(ABC):
    def __init__(self):
        self.__logger = logger.getChild('BoundaryCondition')
    
    @abstractmethod
    def _B_absorbing_boundary(self, positions):
        pass
        
    @abstractmethod
    def _B_reflecting_boundary(self, positions):
        pass

    def __call__(self, positions):
        self.__logger.debug('Checking boundary conditions')
        to_delete = self._B_absorbing_boundary(positions)
        to_update = self._B_reflecting_boundary(positions)
        return to_delete, to_update

INITIAL_SIZE = 100
class Process(ABC, NormalsRG):
    def __init__(self, variables, time_step, seed,
                 force_function=None, force_strength=0,
                 force_cutoff_distance=0):
        NormalsRG.__init__(self, int(1e7), default_size=(1, 2), seed=seed)

        self.__variables = variables
        for var, (dim, dtype) in variables.items():
            if dim == 1:
                shape = (INITIAL_SIZE,)
            else:
                shape = (INITIAL_SIZE, dim)
            self.__dict__['_'+var] = np.empty(shape, dtype=dtype)
        self._current_size = INITIAL_SIZE
        self._active = np.zeros((INITIAL_SIZE,), dtype=bool)
        self._particle_ids = np.empty((INITIAL_SIZE,), dtype=int)
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
        self._process_step()
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

class ParticleType:
    def __init__(self, name, recorder, process):
        self.sources = {}
        self.sinks = {}

        self.process = process

        self.name = name
        recorder.register_parameter(f'particle_type_{name}', {
            'name': name,
            'process': self.process.parameters,
        })

    def step(self):
        for source in self.sources.values():
            for _ in range(np.random.poisson(lam=source.injection_rate*self.process.time_step)):
                self.process.add_particle(position=source.position)
        self.process.step()

        for sink in self.sinks.values():
            self.process.remove_particles(sink.absorb_particles(self))

    @property
    def positions(self):
        return self.process.positions

    @property
    def time(self):
        return self.process.time

class Source:
    def __init__(self, name, particle_type, position, injection_rate, recorder):
        if name in particle_type.sources:
            raise ValueError(f'Source of name {name} already exists in particle type {particle_type.name}')
        self.particle_type = particle_type
        self.particle_type.sources[name] = self

        self.injection_rate = injection_rate
        self.position = np.array(position)

        recorder.register_parameter(f'source_{name}', {
            'name': name,
            'particle_type': particle_type.name,
            'injection_rate': injection_rate,
            'position': self.position,
        })

class Sink(ABC):
    def __init__(self, name, particle_type):
        if name in particle_type.sinks:
            raise ValueError(f'Sink of name {name} already exists in particle type {particle_type.name}')
        particle_type.sinks[name] = self

        self.name = name

    @abstractmethod
    def absorb_particles(self, particle_type):
        pass

