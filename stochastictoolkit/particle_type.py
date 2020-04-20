import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class ParticleType:
    def __init__(self, name, recorder, process, sources=[], sinks=[]):
        self.name = name

        self.sources = {}
        for source in sources:
            if source.name in self.sources:
                raise ValueError(f'Source of name {source.name} already exists in particle type {name}')
            self.sources[source.name] = source
            source.particle_type = self

        self.sinks = {}
        for sink in sinks:
            if sink.name in self.sinks:
                raise ValueError(f'Sink of name {sink.name} already exists in particle type {name}')
            self.sinks[sink.name] = sink
            sink.particle_type = self

        self.process = process

        recorder.register_parameter(f'particle_type_{name}', {
            'name': name,
            'process': self.process.parameters,
            'sources': {s.name: s.parameters for s in self.sources.values()},
            'sinks': {s.name: s.parameters for s in self.sinks.values()},
        })

    def step(self):
        for source in self.sources.values():
            source._inject_particles()

        self.process.step()

        for sink in self.sinks.values():
            sink._absorb_particles()

    @property
    def positions(self):
        return self.process.positions

    @property
    def time(self):
        return self.process.time

class Source(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def _inject_particles(self):
        pass

    @property
    @abstractmethod
    def parameters(self):
        return {
            'name': self.name,
            'particle_type': self.particle_type.name,
        }

class PointSource(Source):
    def __init__(self, name, position, injection_rate):
        super().__init__(name)
        
        self.injection_rate = injection_rate
        self.position = np.array(position)

    @property
    def parameters(self):
        ret = super().parameters
        ret.update({
            'particle_type': self.particle_type.name,
            'injection_rate': self.injection_rate,
            'position': self.position,
        })
        return ret

    def _inject_particles(self):
        process = self.particle_type.process
        for _ in range(np.random.poisson(lam=self.injection_rate*process.time_step)):
            process.add_particle(position=self.position)

class Sink(ABC):
    def __init__(self, name):
        self.name = name

    @property
    @abstractmethod
    def parameters(self):
        return {
            'name': self.name,
            'particle_type': self.particle_type.name,
        }

    @abstractmethod
    def _absorb_particles(self, particle_type):
        pass

