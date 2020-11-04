'''angular_noise_process.py

Contains the implementation for an process with direction noise, direction drift,
position noise and positon drift (particle-particle interactions).

Classes
-------
AngularNoiseProcessWithAngularDrift: Angular noise process with drift
'''
import numpy as np

from .process import Process

class AngularNoiseProcessWithAngularDrift(Process):
    '''Process subclass describing a stochastic process with direction noise.

    This is a catchall implementation for a process with direction noise and drift, together with
    a normal Brownian process.
    '''

    def __init__(self,
                 time_step,
                 boundary_condition,
                 angular_diffusion_coefficient,
                 position_diffusion_coefficient,
                 drift_strength,
                 speed,
                 drift_function=None,
                 force=None,
                 seed=None,
                 alignment_when_reflecting=False):
        '''Initialize the process

        Parameters
        ----------
        time_step: float
            The time step size of the process
        boundary_condition: BoundaryCondition
            The BoundaryCondition subclass object describing the domain's boundary conditions
        angular_diffusion_coefficient: float
            The direction noise strength (angular diffusion coefficient)
        position_diffusion_coefficient: float
            The position diffusion coefficient
        drift_strength: float
            The strength of the direction drift term
        speed: float
            The constant speed of the particles
        drift_function: Function
            A function accepting a position argument returning the drift term
        force: InteractionForce or None
            The particle-particle interaction force. [default: None]
        seed: int or None
            The seed of the normal random number generator
        '''
        # This needs two variables - position and angle (direction)
        variables = {
            'position': (2, float),
            'angle': (1, float),
        }
        super().__init__(variables, time_step, boundary_condition, seed, force=force)

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

        self.__alignment_when_reflecting = alignment_when_reflecting
        
        self.time = 0
        
    @property
    def parameters(self):
        '''Parameters of the process'''
        ret = super().parameters
        ret.update({
            'process': 'AngularNoiseProcessWithAngularDrift',
            'time_step': self.__time_step,
            'position_diffusion_coefficient': self.__position_diffusion_coefficient,
            'angular_diffusion_coefficient': self.__angular_diffusion_coefficient,
            'drift_strength': self.__drift_strength,
            'drift_function': str(self.__drift_function),
            'speed': self.__speed,
            'alignment_when_reflecting': self.__alignment_when_reflecting,
        })
        return ret
        
    def _process_step(self):
        '''Process position (and direction) update method'''
        # Get positions and angles of active particles
        positions = self._position[self._active]
        angles = self._angle[self._active]
        
        # Calculate velocity vectors from angles
        velocities = np.exp(1j*(angles)).view(np.float).reshape(-1, 2)
        
        # Calculate drift term
        if self.__drift_function is not None:
            drift = self.__drift_strength_dt*self.__drift_function(positions, velocities, self.time)
        else:
            drift = 0.
        # Calculate angular diffusion
        diffusion = self.__angular_stepsize*self._normal(size=(self._N_active,))
        self._angle[self._active] += drift + diffusion
        
        # Calculate positional drift
        pos_drift = self._pairwise_force_term(positions) + velocities*self.__speeddt
        # Calculate positional diffusion
        pos_diffusion = self.__pos_stepsize*self._normal(size=(self._N_active, 2))
        return positions + pos_drift + pos_diffusion

    def _reflect_particles(self, to_reflect_a, new_positions, crossing_points, normal_vectors):
        # Calculate vector pointing from crossing point to new position
        d = new_positions - crossing_points
        # Dot product to get the component normal to the boundary
        dotp = (d*normal_vectors).sum(axis=1)
        # The reflected position is then this component twice subtracted from the new (invalid) position
        # Don't update the positions yet, we need to return them (other BCs might need evaluation)
        new_positions -= 2*dotp[:, np.newaxis]*normal_vectors

        # Calculate the velocity vectors from the angles (norm = 1)
        velocities = np.exp(1j*(self._angle[to_reflect_a])).view(np.float).reshape(-1, 2)
        if not self.__alignment_when_reflecting:
            # The new velocity is the reflection of the old velocity at the boundary
            velocities -= 2*(velocities*normal_vectors).sum(axis=1)[:, np.newaxis]*normal_vectors
        else:
            # The new velocity has no component normal to the boundary
            velocities -= (velocities*normal_vectors).sum(axis=1)[:, np.newaxis]*normal_vectors
        # Update the angle
        self._angle[to_reflect_a] = np.arctan2(*velocities.T[::-1])
        # We could have operated on the angles alone (without calculating the velocity vector),
        # but this is more readable and has negligible performance penalty.

        return new_positions
        
    def add_particle(self, position, angle=None):
        '''Add particle at given position and with given direction angle'''
        if angle is None:
            # If no angle was given, generate a random one.
            angle = 2*np.pi*np.random.uniform()
        # Add particle
        super().add_particle(position=position, angle=angle)
        
    @property
    def positions(self):
        '''Current particle positions'''
        return self._position[self._active, :]
    
    @property
    def velocities(self):
        '''Current particle angles'''
        return self.__speeddt/self.__time_step*np.exp(1j*self._angle[self._active]).view(np.float).reshape(-1, 2)
