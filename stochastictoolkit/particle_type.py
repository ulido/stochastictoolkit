import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

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

