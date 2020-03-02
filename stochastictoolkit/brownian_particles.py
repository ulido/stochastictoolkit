import numpy as np
from tqdm import tqdm_notebook as tqdm
import heapq
from collections import namedtuple
import randomgen
import pandas as pd
import pickle

from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)

class Recorder:
    def __init__(self, filename):
        self._parameters = {}
        self._frozen = False
        self._recording_types = {}
        self._recording_types_under_construction = {}

        self._filename = filename
        
        self.__logger = logger.getChild('Recorder')
        
    def register_parameter(self, name, value):
        if self._frozen:
            raise RuntimeError("Trying to register parameter after recording has started!")
        if name in self._parameters:
            raise KeyError(f"Parameter {name} was already registered!")
        self.__logger.info(f"Registering parameter {name}")
        self._parameters[name] = value

    def register_parameters(self, parameters):
        for k, v in parameters.items():
            self.register_parameter(k, v)

    def new_recording_type(self, name, fields):
        self.__logger.info(f"Registering recording type {name}")
        if self._frozen:
            raise RuntimeError("Trying to create type after recording has started!")            
        self._recording_types_under_construction[name] = fields

    def _build_recording_types(self):
        self.__logger.info(f"Building recording types")
        self._frozen = True
        for name, fields in self._recording_types_under_construction.items():
            self._recording_types[name] = (namedtuple(name, list(fields)), [])

    def record(self, type_name, **items):
        if not self._frozen:
            self._build_recording_types()
        rec_type, rows = self._recording_types[type_name]
        self.__logger.info(f"Recording event of type {type_name}")
        rows.append(rec_type(**items))

    def save(self, filename=None):
        if filename is None:
            filename = self._filename
        if not self._frozen:
            self._build_recording_types()
        df = pd.DataFrame([{'parameter': k,
                            'value': str(repr(v)),
                            'pickled_value': str(pickle.dumps(v))}
                           for k, v in self._parameters.items()])
        df.to_hdf(filename, mode='w', key='parameters')
        for type_name, (_, rows) in self._recording_types.items():
            self.__logger.info(f"Saving {len(rows)} recorded events for type {type_name}")
            df = pd.DataFrame(rows)
            df.to_hdf(filename, mode='a', key=type_name)

class NormalsRG:
    def __init__(self, N_normals, default_size=(1,), seed=None, recorder=None):
        self.__N_rng = randomgen.Generator(randomgen.Xoroshiro128(seed, mode='sequence'))
        
        self.__N_normals = N_normals
        self.__N_index = np.inf
        self.__N_default_size = default_size
        self.__N_default_N = np.prod(default_size)

        if recorder is not None:
            recorder.register_parameter('_N_normals', N_normals)
        
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

class EulerMaruyamaNP(NormalsRG):
    def __init__(self, time_step, diffusion_coefficient,
                 boundary_condition=None, seed=None, recorder=None):
        NormalsRG.__init__(self, int(1e7), default_size=(1, 2), seed=seed)
        self.__time_step = time_step
        self.__stepsize = (2*diffusion_coefficient*time_step)**0.5
        self.__boundary_condition = boundary_condition

        self.__positions = np.empty((100, 2), dtype=float)
        self.__active = np.zeros((self.__positions.shape[0],), dtype=bool)
        self.__N_active = 0
        self.__stale_indices = [i for i in range(self.__positions.shape[0])]
        self.time = 0
        
        self.__logger = logger.getChild('EulerMaruyamaNP')
        
    #@profile
    def step(self):
        #self.__logger.debug('Performing EM step')
        if self.__N_active > 0:
            if self.__boundary_condition is None:
                self.__positions[self.__active, :] += self.__stepsize*self._normal(size=(self.__N_active, 2))
            else:
                new_pos = self.__positions[self.__active, :] + self.__stepsize*self._normal(size=(self.__N_active, 2))
                to_delete, to_update = self.__boundary_condition(new_pos)
                to_update_a = np.where(self.__active)[0][to_update]
                self.__positions[to_update_a, :] = new_pos[to_update, :]
                self.remove_particles(to_delete)
        self.time += self.__time_step
        
    def add_particle(self, position):
        self.__logger.info('Adding particle...')
        try:
            idx = heapq.heappop(self.__stale_indices)
            self.__logger.debug('...with index %d' % idx)
        except IndexError:
            old_size = self.__positions.shape[0]
            self.__positions.resize((old_size+100, 2))
            self.__active.resize((old_size+100,))
            self.__stale_indices.extend(i for i in range(old_size+1, old_size+100))
            idx = old_size
            self.__logger.debug('...with new index %d (after extending arrays)' % idx)
        self.__positions[idx] = position
        self.__active[idx] = True
        self.__logger.debug("Active index is %s" % repr(self.__active))
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
                
class ParticleType(EulerMaruyamaNP):
    def __init__(self, name, diffusion_coefficient, time_step, boundary_condition, recorder):
        self.sources = {}
        self.sinks = {}
        
        EulerMaruyamaNP.__init__(self, time_step, diffusion_coefficient=diffusion_coefficient,
                                 boundary_condition=boundary_condition, seed=None)

        self.name = name
        self.time_step = time_step
        recorder.register_parameter(f'particle_type_{name}', {
            'name': name,
            'diffusion_coefficient': diffusion_coefficient,
            'time_step': time_step,
        })

    def step(self):
        for source in self.sources.values():
            for _ in range(np.random.poisson(lam=source.injection_rate*self.time_step)):
                self.add_particle(source.position)
        EulerMaruyamaNP.step(self)

        for sink in self.sinks.values():
            self.remove_particles(sink.absorb_particles(self))

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

