'''process.py

Classes
-------
Process - base class for all stochastic processes
'''

import numpy as np
from abc import ABC, abstractmethod
from quadtree import QuadTree
import itertools
import heapq

from .normalsrg import NormalsRG

__all__ = ['Process']

# This value controls the initial allocation size of variable arrays.
# i.e. the number of particles that can be held initially before we need
# to resize arrays.
PROCESS_VARIABLE_INITIAL_SIZE = 100

class Process(ABC, NormalsRG):
    '''Base class for all stochastic processes.

    All stochastic processes inherit from this Process base class. It provides
    * the stepping infrastructure,
    * normal-distributed random numbers by inheriting from `NormalsRG`,
    * infrastructure for adding and removing particles,
    * infrastructure to calculate pairwise interaction forces

    When subclassing, the following abstract methods need to be implemented:
    * `_process_step`: process-specific stepping code
    * `_reflect_particles`: process-specific code for reflecting particles at boundaries
    * `parameters`: property returning the process parameters

    In addition, a subclass of `Process` needs to call Process.__init__ with a dictionary
    of variable names and a tuple of (dimension, dtype). This dictionary needs to at least
    contain a single variable name 'position'.

    '''
    
    def __init__(self, variables, time_step, boundary_condition, seed, force=None):
        '''Initialize Process.

        Parameters
        ----------
        variables: dict
            Dictionary containing particle variables as keys and a tuple (dimension, numpy
            data type) as value
        time_step: float
            Process time step value
        boundary_condition: BoundaryCondition
            An object subclassing `BoundaryCondition`, implementing the domain's boundaries
        seed: int
            Random number generator seed for NormalsRG !!WILL PROBABLY BE REMOVED!!
        force: InteractionForce
            An object subclassing `InteractionForce` or `None` [default: `None`], implementing
            particle-particle interactions.
        '''
        # Initialize the normal-distributed random numbers generator
        NormalsRG.__init__(self, int(1e7), default_size=(1, 2), seed=seed)
        
        self._boundary_condition = boundary_condition

        # At least position needs to be in variables
        if 'position' not in variables:
            raise ValueError('Subclasses of Process need to specify a position variable!')
        # Initialize the variables with their respective dimensions and dtypes
        self.__variables = variables
        for var, (dim, dtype) in variables.items():
            # Treat scalars differently
            if dim == 1:
                shape = (PROCESS_VARIABLE_INITIAL_SIZE,)
            else:
                shape = (PROCESS_VARIABLE_INITIAL_SIZE, dim)
            # Insert in class `__dict__` with underscore for obfuscation
            self.__dict__['_'+var] = np.empty(shape, dtype=dtype)
        # Initialize reshaping infrastructure
        self._current_size = PROCESS_VARIABLE_INITIAL_SIZE
        self._active = np.zeros((PROCESS_VARIABLE_INITIAL_SIZE,), dtype=bool)
        self._force = np.zeros((PROCESS_VARIABLE_INITIAL_SIZE, self._position.shape[1]), dtype=float)
        self._inst_velocity = np.zeros((PROCESS_VARIABLE_INITIAL_SIZE, self._position.shape[1]), dtype=float)
        self._particle_ids = np.empty((PROCESS_VARIABLE_INITIAL_SIZE,), dtype=int)
        self._N_active = 0
        self._stale_indices = [i for i in range(self._active.shape[0])]

        self.force = force

        # Counter for particle ids
        self._particle_counter = itertools.count()
        
        self.time_step = time_step
        self.time = 0

    def step(self):
        '''Perform one time step of the process.'''
        # Only do work if we actually have particles in the domain
        if self._N_active > 0:
            # Step particle positions (this is the specific process step function, e.g. Brownian)
            new_positions = self._process_step()
            # Calculate instantaneous velocity (disregarding reflection!)
            self._inst_velocity[self._active, :] = (new_positions - self._position[self._active, :])/self.time_step

            bc = self._boundary_condition
            # Evaluate boundary conditions
            # All of the returns from the boundary conditions are
            # boolean arrays.  Any of them can be None when there are
            # no boundaries of the given type.
            
            # Reflect particles
            to_reflect = bc.reflecting_boundary(new_positions)
            if (to_reflect is not None) and to_reflect.any():
                # Get the indices of active particles
                aidx = np.where(self._active)[0]
                # From this get the indices of the particles which need to be reflected
                to_reflect_a = aidx[to_reflect]

                # Check if we want to do true reflection or simply disallow updates
                if self._boundary_condition.true_reflection:
                    # Calculate crossing points and normal vectors for each reflected particle
                    crossing_points, normal_vectors = (
                        bc.get_reflective_crossing_and_normal(self._position[to_reflect_a, :],
                                                              new_positions[to_reflect, :]))
                    # Reflect particles (this is a process-specific function again because other
                    # variables other than position might be affected too (such as e.g. velocity)
                    new_positions[to_reflect, :] = self._reflect_particles(to_reflect_a, new_positions[to_reflect, :],
                                                                           crossing_points, normal_vectors)
                else:
                    # Discard new positions otherwise
                    new_positions[to_reflect, :] = self._position[to_reflect_a, :]

            # Periodic boundaries
            to_periodic = bc.periodic_boundary(new_positions)
            if (to_periodic is not None) and to_periodic.any():
                # Get the indices of active particles
                aidx = np.where(self._active)[0]
                # From this get the indices of the particles whose positions need to be adjusted
                to_periodic_a = aidx[to_periodic]

                # And update the relevant positions
                new_positions[to_periodic, :] = bc.get_periodic_new_position(new_positions[to_periodic, :])

            self._position[self._active, :] = new_positions

            # Delete any absorbed particles
            to_delete = bc.absorbing_boundary(new_positions)
            if to_delete is not None:
                self.remove_particles(to_delete)
        
        # Update time
        self.time += self.time_step

    @abstractmethod
    def _process_step(self):
        '''Process-specific function to perform a time step.'''
        pass

    @abstractmethod
    def _reflect_particles(self, to_reflect_a, new_positions, crossing_points, tangent_vectors):
        '''Process-specific function to reflect particle positions at boundaries.'''
        pass

    def _pairwise_force_term(self, positions):
        '''Process infrastructure to calculate pairwise interaction forces.

        Currently only works in 2D (uses a quadtree internally).
        '''
        # Don't do anything if no force was given
        if self.force is None:
            return 0

        force_obj = self.force
        forces = np.empty_like(positions)

        # Calculate any ghost positions due to periodic boundaries
        ghost_positions = self._boundary_condition.periodic_ghost_positions(positions, force_obj.cutoff_distance)

        # We use a quadtree to search particle neighborhoods within a cutoff distance
        # Initialize the quadtree neighborhood search domain
        mi, ma = positions.min(), positions.max()
        if ghost_positions is not None and ghost_positions.shape[0] > 0:
            mi = min(mi, ghost_positions.min())
            ma = max(ma, ghost_positions.max())
        
        ce = (ma+mi)/2
        hd = max((ma-mi)/1.99, 1e-5)
        qt = QuadTree([ce, ce], hd)

        # Insert all particle positions into the quadtree
        qt.insert_points(positions)
        if ghost_positions is not None:
            qt.insert_points(ghost_positions)
        # For each particle, search its neighborhood and calculate the overall force on it
        for i, q in enumerate(qt.query_self(force_obj.cutoff_distance)[:forces.shape[0]]):
            forces[i] = force_obj(q-positions[i][np.newaxis]).sum(axis=0)

        # Save the instantaneous forces (so we can output them if needed)
        self._force[self._active, :] = forces

        # Return the force times the time step
        return self.time_step*forces

    def add_particle(self, **kwargs):
        '''Add a particle to the domain.

        All keyword arguments need to be valid variable names for the Process subclass.
        No checking is done if every variable is initialized correctly, therefore this method
        should be overridden by the Process subclass and called only from there.'''

        # Check if a stale index exists in the heap
        try:
            # If yes, use the first one
            idx = heapq.heappop(self._stale_indices)
        except IndexError:
            # If not, resize all the variable arrays
            old_size = self._current_size
            new_size = old_size + 100 # This should probably be doubling, instead of adding an arbitrary amount.
            for var, (dim, _) in self.__variables.items():
                if dim == 1:
                    self.__dict__['_' + var].resize((new_size,))
                else:
                    self.__dict__['_' + var].resize((new_size, dim))
            self._active.resize((new_size,))
            self._force.resize((new_size, self._force.shape[1]))
            self._inst_velocity.resize((new_size, self._inst_velocity.shape[1]))
            self._particle_ids.resize((new_size,))
            self._stale_indices.extend(i for i in range(old_size+1, new_size))
            self._current_size = new_size
            idx = old_size

        # Initialize all given variables
        for var, value in kwargs.items():
            self.__dict__['_'+var][idx] = value
        # Set this index to active
        self._active[idx] = True
        self._force[idx] = 0
        # Initialize particle id
        self._particle_ids[idx] = next(self._particle_counter)
        self._N_active += 1

    def remove_particles(self, indexes):
        '''Remove particles with the given indices.'''
        # Do nothing if no indices given
        if len(indexes) == 0:
            return
        # Figure out which system indices correspond to the given user indices
        indexes = np.where(self._active)[0][indexes]
        # Set these indices to stale and push to the heap
        for index in indexes:
            heapq.heappush(self._stale_indices, index)
        # Set these indices to inactive
        self._active[indexes] = False
        self._N_active -= len(indexes)

    @property
    @abstractmethod
    def parameters(self):
        '''Process parameters

        A subclass needs to override this function and retrieve the Process parameters from Process.parameters.
        '''
        return {
            'force': self.force.parameters if self.force is not None else None,
            'boundary_condition': self._boundary_condition.parameters,
        }

    @property
    def particle_ids(self):
        '''The unique particle IDs.'''
        return self._particle_ids[self._active]

    @property
    def forces(self):
        '''The instantaneous particle interaction forces.'''
        return self._force[self._active]

    @property
    def instantaneous_velocity(self):
        '''The instanteneous particle velocities.'''
        return self._inst_velocity[self._active]
