from EMsimul import ParticleType, Source, ReceptorCollection, BoundaryCondition, Recorder
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, Point, MultiPolygon, LineString, MultiLineString
from shapely.ops import split
from shapely.vectorized import contains as shapely_contains

import time

class WorldBoundaryCondition(BoundaryCondition):
    '''Boundary condition to describe the world for molecules
    Accepts the following arguments:
      - cell_position: Reference to array holding the current cell position - this needs to be shared and updated outside of WorldBoundaryCondition
      - boundary_shape: Shapely geometry object that describes the boundary
      - outer_radius: Float that defines the radius of the absorbing outer boundary (default value: 10)
    '''
    def __init__(self, recorder, cell_position, boundary_shape, outer_radius=10):
        # Initialize the super
        BoundaryCondition.__init__(self)
        # Initialize the parameters
        self.__outer_radius = outer_radius
        self.__cell_position = cell_position
        self.__boundary_shape = boundary_shape

        recorder.register_parameter('outer_radius', outer_radius)
        
    def _B_absorbing_boundary(self, positions):
        # Calculate the distance of the molecules from the origin
        radii = np.linalg.norm(positions, axis=1)
        # Return the indices of particles that are outside of the outer absorbing radius
        return np.where(radii > self.__outer_radius)[0]

    def _B_reflecting_boundary(self, positions):
        # Calculate the distance from the cell
        distance = np.linalg.norm(positions - self.__cell_position[np.newaxis], axis=1)
        # Return a bool array, true for any particle whose new position is not outside the domain
        return ~((distance < 1) | shapely_contains(self.__boundary_shape, *positions.T))

class DirectionFindingCell:
    def __init__(self, position, particle_types, boundary_shape, recorder,
                 N_receptors=12,
                 memory_decay_rate=0.01, saturation_threshold=3, cell_time_step=1e-2, cell_speed=1):

        # Initialize the receptor collection
        self.__receptor_collection = ReceptorCollection('ReceptorA', position, particle_types[0],
                                                        N_receptors=N_receptors, recorder=recorder)

        # Initialize parameters and boundary
        self.__memory_decay_rate = memory_decay_rate
        self.__saturation_threshold = saturation_threshold
        self.__lookback_time = 4.0/memory_decay_rate
        self.__boundary_shape = boundary_shape.buffer(1)
        self.__cell_time_step = cell_time_step
        self.__cell_speed = cell_speed

        # Record parameters
        self.__recorder = recorder
        recorder.register_parameter('memory_decay_rate', self.__memory_decay_rate)
        recorder.register_parameter('saturation_threshold', self.__saturation_threshold)
        recorder.register_parameter('lookback_time', self.__lookback_time)
        recorder.register_parameter('cell_time_step', self.__cell_time_step)
        recorder.register_parameter('cell_speed', self.__cell_speed)

        # Set up type to record cell positions
        recorder.new_recording_type('CellPosition', ['time', 'position_x', 'position_y', 'velocity_x', 'velocity_y'])
        
        # Initialize state and position variables
        self.__cell_next_time = 0
        self.__position = position

        # Cell cannout be outside of the domain - check!
        if self.__boundary_shape.contains(Point(self.__position)):
            raise RuntimeError("Cell start position can't be outside of domain!")
        
    def __activation(self, time):
        receptor_collection = self.__receptor_collection
        # Array to hold current activation values
        activation = np.zeros((receptor_collection.N_receptors,), dtype=float)
        # Calculate activation for each receptor
        for r in range(activation.shape[0]):
            # Initialize to zero
            a = 0
            # Only go back to lookback time
            stop = time-self.__lookback_time
            # Iterate over all recent receptor absorption events in reverse order
            # (from latest to oldest)
            for tau in receptor_collection.receptor_absorptions[r][::-1]:
                # Stop iteration when tau is older than the lookback time
                if tau < stop:
                    break
                # Add each event to the activation, with an exponential memory decay
                a += np.exp(-self.__memory_decay_rate*(time-tau))
            # Activations saturate to a value between 0 and 1, using a tanh with a saturation threshold
            activation[r] = np.tanh(a/self.__saturation_threshold)
        return activation
    
    def __direction(self, time):
        # Calculate the receptor activations
        activation = self.__activation(time)
        # The direction/velocity is then given by the weighted average of the receptor positions
        return (self.__receptor_collection.receptor_positions*activation[:, np.newaxis]).sum(axis=0)/activation.shape[0]  #np.clip(activation.sum(), 1, None)

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
            velocity = self.__direction(time)
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

class World:
    def __init__(self, filename, initial_cell_position, source_position, boundary_shape,
                 N_receptors=3, memory_decay_rate=0.01, saturation_threshold=3,
                 outer_radius=10, injection_rate=10, time_step=1e-4,
                 cell_speed=0.01, cell_time_step=1e-2):
        # Initialize object to record events, parameters and positions
        recorder = Recorder(filename)
        self.recorder = recorder

        # Calculate the average number of particles appearing per time step (Poisson distribution lambda)
        self.injection_lam = injection_rate*time_step

        # State variable to check if the cell has hit the source
        self.cell_has_hit_source = False

        # Initialize simulation variables
        self.cell_position = np.array(initial_cell_position, dtype=float)
        self._cell_next_time = 0

        # Set up boundary condition
        boundary_condition = WorldBoundaryCondition(recorder,
                                                    self.cell_position,
                                                    boundary_shape=boundary_shape,
                                                    outer_radius=outer_radius)
        # Initialize the molecule stepper
        A = ParticleType('A', 1, time_step, boundary_condition, recorder)
        self.particle_types = [A]
        self.sources = [Source('SourceA', A, source_position, injection_rate, recorder)]

        # Initialize the cell
        self.cell = DirectionFindingCell(self.cell_position, self.particle_types, boundary_shape,
                                         N_receptors=N_receptors, recorder=recorder,
                                         memory_decay_rate=memory_decay_rate, saturation_threshold=saturation_threshold,
                                         cell_time_step=cell_time_step, cell_speed=cell_speed)

        # Register parameters
        recorder.register_parameters({
            'boundary_shape': boundary_shape,
        })
        # Set up the data type to record the final molecule positions
        recorder.new_recording_type('FinalMoleculePositions', ['particle_type', 'x', 'y'])

    def step(self):
        # Step the molecule positions
        for pt in self.particle_types:
            pt.step()

        # Step the cell position
        self.cell.step(self.time)
        # Check if the cell has hit the source
        if np.linalg.norm(self.cell.position - self.sources[0].position) < 1.5:
            self.cell_has_hit_source = True
        
    def run(self, for_time, tqdm_position=0):
        # Calculate the stopping time
        end_time = self.time + for_time
        last_wall_time = time.time()
        try:
            with tqdm(total=for_time,
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                      unit='time unit',
                      smoothing=0.01) as pbar:
                # Step until the cell has hit the source or the end time has been reached
                while self.time < end_time:
                    old_time = self.time
                    self.step()
                    pbar.update(self.time - old_time)
                    if self.cell_has_hit_source:
                        break
                    if time.time() - last_wall_time > 30:
                        self.recorder.save()
                        last_wall_time = time.time()
            # Record the final positions of the molecules
            for pt in self.particle_types:
                for p in pt.positions:
                    self.recorder.record('FinalMoleculePositions', particle_type=pt.name, x=p[0], y=p[1])
        finally:
            self.recorder.save()

    @property
    def molecule_positions(self):
        return {pt.name: pt.positions.copy() for pt in self.particle_types}
    
    @property
    def time(self):
        return self.particle_types[0].time
    
if __name__ == '__main__':
    from multiprocessing import Pool
    import traceback

    def run_func(args):
        def wedge(angle=np.pi/8):
            c = np.cos(angle)
            s = np.sin(angle)
            return 50*np.array([(c, -s), (0, 0), (c, s)])

        wedge_coo1 = wedge(np.pi/8)
        wedge_coo1[:, 0] = 2 + wedge_coo1[:, 0]
        wedge1 = Polygon(wedge_coo1)
        wedge_coo2 = wedge(np.pi/8)
        wedge_coo2[:, 0] = -2 - wedge_coo2[:, 0]
        wedge2 = Polygon(wedge_coo2)

        boundary_shape = MultiPolygon([wedge1, wedge2])

        i, mu = args
        w = World(f'dirfinding{i+60:05}.h5', initial_cell_position=[10, 8], source_position=[-10, -6],
                  N_receptors=12, boundary_shape=boundary_shape,
                  cell_speed=1, injection_rate=10, outer_radius=30, memory_decay_rate=mu)
        w.run(1000)

    def try_run(args):
        try:
            run_func(args)
        except Exception as e:
            traceback.print_exc()

    source_distances = [0.05] #np.linspace(1.1, 20, 10)[[0]]
    with Pool(20) as p:
        tasks = [L for L in source_distances for i in range(20)]
        p.map(try_run, enumerate(tasks))
        #for _ in tqdm(p.imap_unordered(run_func, enumerate(tasks)), total=len(tasks)):
        #    pass
