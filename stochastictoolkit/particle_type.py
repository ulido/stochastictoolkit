import numpy as np
import heapq
from collections import namedtuple
import randomgen
import pandas as pd
import pickle
import itertools
import tables
import pathlib

from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)

class Recorder:
    def __init__(self, filename):
        self._parameters = {}
        self._frozen = False
        self._recording_types = {}
        self._recording_types_under_construction = {}
        self._arrays_to_save = {}

        if pathlib.Path(filename).exists():
            raise ValueError(f'File {filename} exists!')
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

    def record_array(self, name, array, index_value=None):
        # Saving arrays is done instantaneously rather than waiting until save is called.
        # Otherwise we might run into memory issues.
        with tables.open_file(self._filename, mode='a') as h5file:
            if name not in h5file.root:
                data = h5file.create_earray(f'/{name}', 'data', obj=array[np.newaxis], createparents=True)
                print(data)
                if index_value is not None:
                    index = h5file.create_earray(f'/{name}', 'index', obj=np.array(index_value)[np.newaxis])
            else:
                group = h5file.get_node(f'/{name}')
                if 'index' in group:
                    if index_value is None:
                        raise ValueError('Index present but no index value specified!')
                    group['index'].append(np.array(index_value)[np.newaxis])
                    group['data'].append(array[np.newaxis])
            
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

    def save(self):
        if not self._frozen:
            self._build_recording_types()
        df = pd.DataFrame([{'parameter': k,
                            'value': str(repr(v)),
                            'pickled_value': str(pickle.dumps(v))}
                           for k, v in self._parameters.items()])
        df.to_hdf(self._filename, mode='a', key='parameters')
        for type_name, (_, rows) in self._recording_types.items():
            self.__logger.info(f"Saving {len(rows)} recorded events for type {type_name}")
            df = pd.DataFrame(rows)
            df.to_hdf(self._filename, mode='a', key=type_name)

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

class Process(ABC, NormalsRG):
    def __init__(self, variables, time_step, seed, initial_size=100):
        NormalsRG.__init__(self, int(1e7), default_size=(1, 2), seed=seed)

        self.__variables = variables
        for var, (dim, dtype) in variables.items():
            if dim == 1:
                shape = (initial_size,)
            else:
                shape = (initial_size, dim)
            self.__dict__['_'+var] = np.empty(shape, dtype=dtype)
        self._current_size = initial_size
        self._active = np.zeros((initial_size,), dtype=bool)
        self._particle_ids = np.empty((initial_size,), dtype=int)
        self._N_active = 0
        self._stale_indices = [i for i in range(self._active.shape[0])]

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
        pass

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

