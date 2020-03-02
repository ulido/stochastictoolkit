from EMSimulation import ParticleType, Source, BoundaryCondition, Recorder
from MovingCellSimulation import ReceptorCollection, DirectionFindingCell

import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.vectorized import contains as shapely_contains

import time

class WorldBoundaries(BoundaryCondition):
    '''Boundary condition to describe the world for molecules
    Accepts the following arguments:
      - cell_position: Reference to array holding the current cell position - this needs to be shared and updated outside of WorldBoundaries
      - boundary_shape: Shapely geometry object that describes the boundary
      - outer_radius: Float that defines the radius of the absorbing outer boundary (default value: 10)
    '''
    def __init__(self, recorder, cell_position, boundary_shape, outer_radius=10):
        # Initialize the super
        super().__init__()
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
        boundary_condition = WorldBoundaries(recorder,
                                             self.cell_position,
                                             boundary_shape=boundary_shape,
                                             outer_radius=outer_radius)
        # Initialize the molecule stepper
        A = ParticleType('A', 1, time_step, boundary_condition, recorder)
        B = ParticleType('B', 1, time_step, boundary_condition, recorder)
        self.particle_types = [A, B]
        self.sources = [Source('SourceA', A, [-5, 4], injection_rate, recorder),
                        Source('SourceB', B, [-20, -12], injection_rate/3, recorder)]

        receptor_collections = [
            ReceptorCollection('ReceptorA',
                               self.cell_position, recorder, A,
                               N_receptors=N_receptors,
                               memory_decay_rate=memory_decay_rate, saturation_threshold=saturation_threshold),
            ReceptorCollection('ReceptorB',
                               self.cell_position, recorder, B,
                               N_receptors=N_receptors,
                               memory_decay_rate=memory_decay_rate, saturation_threshold=saturation_threshold)
        ]

        def velocity(receptor_collections):
            dirA = receptor_collections[0].direction()
            dirB = receptor_collections[1].direction()
            v = dirA + 2*dirB
            vl = np.linalg.norm(v)
            if vl > 0:
                return v * np.tanh(1/vl)
            else:
                return v
        # Initialize the cell
        self.cell = DirectionFindingCell(self.cell_position, self.particle_types, boundary_shape, recorder, receptor_collections,
                                         velocity=velocity, cell_time_step=cell_time_step, cell_speed=cell_speed)

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
        if np.linalg.norm(self.cell.position - self.sources[-1].position) < 1.5:
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
        w = World(f'multidiffsource{i:05}.h5', initial_cell_position=[20, 16], source_position=[-10, -6],
                  N_receptors=12, boundary_shape=boundary_shape,
                  cell_speed=1, injection_rate=10, outer_radius=30, memory_decay_rate=mu)
        w.run(2000)

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
