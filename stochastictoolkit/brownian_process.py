'''brownianprocess.py

Contains the implementation for a Brownian process.

Classes
-------
BrownianProcess: Stochastic process for simulation Brownian motion
'''

import numpy as np

from .process import Process

__all__ = ['BrownianProcess']

class BrownianProcess(Process):
    '''Process subclass describing a Brownian process

    This performs simple Brownian motion for the `ParticleType` it is assigned to. 
    '''
    def __init__(self,
                 time_step,
                 diffusion_coefficient,
                 boundary_condition,
                 ndim=2,
                 external_drift=None,
                 interaction_force=None,
                 seed=None):
        '''Initialize the Brownian process

        Parameters
        ----------
        time_step: float
            Time step size
        diffusion_coefficient: float
            The diffusion coefficient of the particles
        boundary_condition: BoundaryCondition
            The BoundaryCondition subclass object describing the domain's boundary conditions
        ndim: int
            Number of dimensions (default: 2)
        external_drift: ExternalDrift or None
            The external drift. [default: None]
        interaction_force: InteractionForce or None
            The particle-particle interaction force. [default: None]
        seed: int or None
            The seed of the normal random number generator
        '''
        # Brownian particles are completely described by only their position.
        variables = {
            'position': (ndim, float)
        }
        # Initialize the Process superclass
        super().__init__(variables, time_step, boundary_condition, seed, force=interaction_force)

        self.__ndim = ndim
        self.__diffusion_coefficient = diffusion_coefficient

        self.__external_drift = external_drift
        
        # The step size is $\sqrt{2D\Delta t}$
        self.__stepsize = (2*diffusion_coefficient*time_step)**0.5

    @property
    def parameters(self):
        '''The parameters of the Brownian motion'''
        ret = super().parameters
        ret.update({
            'process': 'BrownianProcess',
            'time_step': self.time_step,
            'diffusion_coefficient': self.__diffusion_coefficient,
            'dimensions': self.__ndim,
            'external_drift': self.__external_drift.parameters if self.__external_drift is not None else None,
        })
        return ret
    
    def _process_step(self):
        '''Brownian motion position update method'''
        # Get positions of active particles
        positions = self._position[self._active, :]
        # Calculate the drift (from the interaction forces)
        drift = self._pairwise_force_term(positions)
        # Calculate the external drift
        if self.__external_drift is not None:
            drift += self.time_step*self.__external_drift(positions)
        
        # Calculate the diffusion term
        diffusion = self.__stepsize*self._normal(size=(self._N_active, self.__ndim))
        # Return new positions
        return positions + drift + diffusion

    def _reflect_particles(self, to_reflect_a, new_positions, crossing_points, tangent_vectors):
        '''Reflect particles at reflecting boundaries'''
        # Vector from crossing point to new position
        d = new_positions - crossing_points
        # Component of this vector in the direction of the boundary normal
        dotp = (d*tangent_vectors).sum(axis=1)
        # The resulting new position is the crossing point plus twice this component
        return crossing_points-d+2*dotp[:, np.newaxis]*tangent_vectors

    @property
    def positions(self):
        '''Current particle positions'''
        return self._position[self._active, :]
