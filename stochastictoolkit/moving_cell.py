from .brownian_particles import Sink

import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import split

import logging

logger = logging.getLogger(__name__)

import inspect

class ReceptorCollection(Sink):
    def __init__(self, name, position, recorder, particle_type, receptor_size=0.05, N_receptors=3, memory_decay_rate=0.01, saturation_threshold=3):
        self.__position = position
        self.N_receptors = N_receptors
        self.__receptor_size = receptor_size
        self.receptor_positions = np.array([(np.cos(phi), np.sin(phi)) for phi in np.arange(0, 2*np.pi, 2*np.pi/N_receptors)])
        self.__memory_decay_rate = memory_decay_rate
        self.__saturation_threshold = saturation_threshold
        self.__lookback_time = 4.0/memory_decay_rate

        Sink.__init__(self, name, particle_type)

        self.__particle_type = particle_type
        
        self.receptor_absorptions = {r: [] for r in range(N_receptors)}
        
        self.__recorder = recorder
        parameters = {'receptor%d_position' % (i+1): x
                      for i, x in enumerate(self.receptor_positions)}
        parameters['receptor_size'] = receptor_size
        parameters['particle_type'] = particle_type.name
        parameters['memory_decay_rate'] = self.__memory_decay_rate
        parameters['saturation_threshold'] = self.__saturation_threshold
        parameters['lookback_time'] = self.__lookback_time
        recorder.register_parameter(f'ReceptorCollection_{name}', parameters)
        recorder.new_recording_type(f'ReceptorReceivedEvents_{name}', ['time', 'receptor'])

        self.__logger = logger.getChild('ReceptorCollection')

    def absorb_particles(self, particle_type):
        self.__logger.debug('Checking if particles need to be bound')

        positions = particle_type.positions
        relcell = positions - self.__position
        candidates = np.linalg.norm(relcell, axis=1) < 1+self.__receptor_size

        to_delete = []
        if candidates.any():
            self.__logger.debug('Find candidates close to cell')
            p = relcell[candidates, :]
            for r, rpos in enumerate(self.receptor_positions):
                close = (np.linalg.norm(p-rpos, axis=1) < self.__receptor_size)
                num_close = close.sum()
                if num_close > 0:
                    time = particle_type.time
                    self.__logger.info('Find %d particles close to receptor %d' % (num_close, r))
                    candidates_idx = np.where(candidates)[0][close]
                    to_delete.extend(candidates_idx)
                    for n in range(num_close):
                        self.receptor_absorptions[r].append(time)
                        self.__recorder.record(f'ReceptorReceivedEvents_{self.name}', time=time, receptor=r)
        return to_delete
        
    def activation(self):
        time = self.__particle_type.time
        # Array to hold current activation values
        activation = np.zeros((self.N_receptors,), dtype=float)
        # Calculate activation for each receptor
        for r in range(activation.shape[0]):
            # Initialize to zero
            a = 0
            # Only go back to lookback time
            stop = time-self.__lookback_time
            # Iterate over all recent receptor absorption events in reverse order
            # (from latest to oldest)
            for tau in self.receptor_absorptions[r][::-1]:
                # Stop iteration when tau is older than the lookback time
                if tau < stop:
                    break
                # Add each event to the activation, with an exponential memory decay
                a += np.exp(-self.__memory_decay_rate*(time-tau))
            # Activations saturate to a value between 0 and 1, using a tanh with a saturation threshold
            activation[r] = np.tanh(a/self.__saturation_threshold)
        return activation

    def direction(self):
        # Calculate the receptor activations
        activation = self.activation()
        # The direction/velocity is then given by the weighted average of the receptor positions
        return (self.receptor_positions*activation[:, np.newaxis]).sum(axis=0)/activation.shape[0]  #np.clip(activation.sum(), 1, None)

class DirectionFindingCell:
    def __init__(self, position, particle_types, boundary_shape, recorder, receptor_collections,
                 velocity=None, cell_time_step=1e-2, cell_speed=1):

        # Initialize the receptor collection
        self.__receptor_collections = receptor_collections

        if velocity is None:
            self.__velocity = lambda receptor_collections: np.mean([rc.direction() for rc in receptor_collections], axis=0)
        else:
            self.__velocity = velocity

        # Initialize parameters and boundary
        self.__boundary_shape = boundary_shape.buffer(1)
        self.__cell_time_step = cell_time_step
        self.__cell_speed = cell_speed

        # Record parameters
        self.__recorder = recorder
        recorder.register_parameter('cell_time_step', self.__cell_time_step)
        recorder.register_parameter('cell_speed', self.__cell_speed)
        recorder.register_parameter('velocity_function', inspect.getsource(self.__velocity))

        # Set up type to record cell positions
        recorder.new_recording_type('CellPosition', ['time', 'position_x', 'position_y', 'velocity_x', 'velocity_y'])
        
        # Initialize state and position variables
        self.__cell_next_time = 0
        self.__position = position

        # Cell cannout be outside of the domain - check!
        if self.__boundary_shape.contains(Point(self.__position)):
            raise RuntimeError("Cell start position can't be outside of domain!")
    
    def __reflect_cell(self, new_position):
        # Reflect cell if the proposed new position crosses a boundary
        # This LineString then intersects the boundary
        l = LineString([self.__position, new_position])

        # If the boundary is a single unbroken line, it is a LineString
        # otherwise it is as MultiLineString containing a list of LineStrings
        boundary = self.__boundary_shape.boundary
        if boundary.type == 'LineString':
            boundary = MultiLineString([boundary])
        if boundary.type != 'MultiLineString':
            raise ValueError(f'Cannot deal with boundary of type {bound.type}!')

        # Iterate over all disconnected parts of the boundary to extract the part we crossed
        for boundary_part in boundary:
            if boundary_part.intersects(l):
                break
        else:
            # This should not happen - we have after all crossed the boundary!
            raise ValueError('Not intersecting boundary!')

        # Split the boundary part on the intersecting line
        split_boundary = split(boundary_part, l)[0]
        # Extract the intersection point
        intersect_point = np.array(split_boundary.coords[-1])
        # Calculate the normalized tangent of the boundary at that point
        tangent = np.diff([split_boundary.coords[-2], intersect_point], axis=0)
        tangent /= np.linalg.norm(tangent)
        # Get the part of the jump that is inside the boundary
        vector_inside = np.array(new_position) - intersect_point
        # Calculate the dot product with the tangent to get the component along the boundary
        # (which is preserved under reflection)
        tangent_component = np.dot(tangent, vector_inside)
        
        # Calculate and return the reflected new position
        return np.squeeze(intersect_point - vector_inside + 2*tangent*tangent_component)

    def step(self, time):
        'Step the cell (if the next time to do so has been reached). Accepts one argument, the current simulation time.'
        # Check if the next step time has been reached
        if time >= self.__cell_next_time:
            # Set the next time we should perform a step
            self.__cell_next_time = time + self.__cell_time_step
            # Calculate the velocity/direction of the step from the receptor activation
            velocity = self.__velocity(self.__receptor_collections)
            # Update the new cell position from the velocity, the cell speed and the time step length
            new_position = self.__position + velocity*self.__cell_speed*self.__cell_time_step
            # Check if we crossed the boundary, reflect the cell if yes
            if self.__boundary_shape.contains(Point(new_position)):
                new_position = self.__reflect_cell(new_position)
            # Set the new position
            self.__position[:] = new_position
            # Record for posterity
            self.__recorder.record('CellPosition', time=time,
                                   position_x=self.__position[0], position_y=self.__position[1],
                                   velocity_x=velocity[0], velocity_y=velocity[1])

    @property
    def position(self):
        return self.__position
