'''particle_type.py

Contains the infrastructure to simulate different particle types.

Classes
-------
ParticleType: Describes a single particle type in a simulation
Source: Base class for particle sources
PointSource: Implementation of a particle point source
Sink: Base class for particle sinks
'''
import numpy as np
from abc import ABC, abstractmethod

__all__ = ['ParticleType', 'Source', 'Sink', 'PointSource']

class ParticleType:
    '''Class describing a type of particle in the computational domain
    
    This is the central class for a particle type, holding the infrastructure and
    information of how a type of particle moves, gets created and removed.
    '''
    def __init__(self, name, process, sources=[], sinks=[], recorder=None):
        '''Initialize a particle type

        Parameters
        ----------
        name: str
            Name of the particle type
        process: Process
            The stochastic process the particle dynamics is described by. Needs to
            be a subclass of Process.
        sources: list
            List of sources in the domain. All entries need to be a subclass of Source.
            Sources all need to have unique names. A source cannot be shared between
            particle types.
        sinks: list
            List of sinks in the domain. All entries need to be a subclass of Sink
            Sinks all need to have unique names. A sink cannot be shared between
            particle types.
        recorder: Recorder
            Recorder object for recording parameter and dynamical information (default: None)
        '''
        self.name = name

        # Register ourselves with the sources
        self.sources = {}
        for source in sources:
            if source.name in self.sources:
                raise ValueError(f'Source of name {source.name} already exists in particle type {name}')
            self.sources[source.name] = source
            source.particle_type = self

        # Register ourselves with the sinks
        self.sinks = {}
        for sink in sinks:
            if sink.name in self.sinks:
                raise ValueError(f'Sink of name {sink.name} already exists in particle type {name}')
            self.sinks[sink.name] = sink
            sink.particle_type = self

        self.process = process

        # Record all the parameters of all components with the recorder object
        if recorder is not None:
            recorder.register_parameter(f'particle_type_{name}', {
                'name': name,
                'process': self.process.parameters,
                'sources': {s.name: s.parameters for s in self.sources.values()},
                'sinks': {s.name: s.parameters for s in self.sinks.values()},
            })

    def step(self):
        '''Perform one time step.'''
        # Inject particles for any sources present
        for source in self.sources.values():
            source._inject_particles()

        # Perform the actual time stepping
        self.process.step()

        # Absorb particles in sinks
        for sink in self.sinks.values():
            sink._absorb_particles()

    @property
    def positions(self):
        '''Current particle positions'''
        return self.process.positions

    @property
    def number_of_particles(self):
        '''Current number of particles'''
        return self.process._N_active

    @property
    def time(self):
        '''Current time'''
        return self.process.time

class Source(ABC):
    '''Base class for all particle sources

    The ParticleType object this source will be assigned to will register
    itself as `self.particle_type`.

    All sources inherit from this class. When subclassing the followin
    abstract methods need to be implemented:
    * `_inject_particles`: inject particles during time stepping
    * `parameters`: property returning the source parameters
    '''
    def __init__(self, name):
        '''Initialize Source

        Parameters
        ----------
        name: str
            The name of the source
        '''
        self.name = name

    @abstractmethod
    def _inject_particles(self):
        '''Inject particles during a time step.

        This function gets called during `ParticleType.step` to inject the
        appropriate number of particles at the appropriate positions. This is to
        be done via the `self.particle_type.process.add_particle` method.
        '''
        pass

    @property
    @abstractmethod
    def parameters(self):
        '''Source parameters

        A subclass needs to override this function and retrieve the Source parameters from `Source.parameters`
        '''
        return {
            'name': self.name,
            'particle_type': self.particle_type.name,
        }

class PointSource(Source):
    '''Implementation of a point source.

    This source will inject particles at the given position with a constant rate.
    '''
    def __init__(self, name, position, injection_rate):
        '''Initialize PointSource

        Parameters
        ----------
        name: str
            Name of the point source
        positions: sequence of length 2
            Position coordinates of the point source
        injection_rate: float
            Particle injection rate
        '''
        super().__init__(name)
        
        self.injection_rate = injection_rate
        self.position = np.array(position)

    @property
    def parameters(self):
        '''PointSource parameters'''
        ret = super().parameters
        ret.update({
            'particle_type': self.particle_type.name,
            'injection_rate': self.injection_rate,
            'position': self.position,
        })
        return ret

    def _inject_particles(self):
        '''Particle injection method'''
        # Inject N number of particles where $N~Poisson(\Delta t \lambda)$
        process = self.particle_type.process
        for _ in range(np.random.poisson(lam=self.injection_rate*process.time_step)):
            process.add_particle(position=self.position)

class Sink(ABC):
    '''Base class for all particle sinks

    All sinks inherit from this class. When subclassing the following abstract
    methods need to be implemented:
    * `_absorb_particles`: Removal of particles due to absorption
    * `parameters`: property returning the sink parameters
    '''
    def __init__(self, name):
        '''Initialize the sink

        Parameters
        ----------
        name: str
            The name of the sink
        '''
        self.name = name

    @property
    @abstractmethod
    def parameters(self):
        '''Sink parameters

        A subclass needs to override this function and retrieve the Sink parameters from `Sink.parameters`
        '''
        return {
            'name': self.name,
            'particle_type': self.particle_type.name,
        }

    @abstractmethod
    def _absorb_particles(self, particle_type):
        '''Particle absorption method

        This is called by the `ParticleType.step` function to remove particles. This could be due to a region of
        the domain being (partially) absorbing, or even removing random particles from the domain. Removal is to be
        done via the `self.particle_type.process.remove_particle` method.
        '''
        pass

