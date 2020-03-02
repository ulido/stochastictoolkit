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
        if boundary_shape.type != 'MultiPolygon':
            self.__boundary_shape = MultiPolygon([boundary_shape])
        else:
            self.__boundary_shape = boundary_shape

        self.__boundary_bounds = [np.array(b.bounds).reshape((2, 2)).T for b in self.__boundary_shape.geoms]

        recorder.register_parameter('outer_radius', outer_radius)
        
    def _B_absorbing_boundary(self, positions):
        # Calculate the distance of the molecules from the origin
        radii = np.linalg.norm(positions, axis=1)
        # Return the indices of particles that are outside of the outer absorbing radius
        return np.where(radii > self.__outer_radius)[0]

    #@profile
    def _B_reflecting_boundary(self, positions):
        # Calculate the distance from the cell
        diff = positions - self.__cell_position[np.newaxis]
        to_update = np.square(diff).sum(axis=1) >= 1
        #distance = np.linalg.norm(positions - self.__cell_position[np.newaxis], axis=1)
        # Return a bool array, true for any particle whose new position is not outside the domain

        x = positions[:, 0]
        y = positions[:, 1]
        for i, bbox in enumerate(self.__boundary_bounds):
            to_update &= ((x < bbox[0, 0]) | (x > bbox[0, 1]) | (y < bbox[1, 0]) | (y > bbox[1, 1]))
            #if inside_bbox.any():
            #    to_update &= ~inside_bbox
#                idx = np.nonzero(inside_bbox)[0]
#                contained = shapely_contains(self.__boundary_shape.geoms[i], *positions[idx, :].T)
#                to_update[idx[contained]] = False
#        else:
#            if shapely_contains(self.__boundary_shape, *positions.T).any():
#                print(repr(positions))
#                raise RuntimeError
        return to_update
        return ~((distance < 1) | shapely_contains(self.__boundary_shape, *positions.T))

class World:
    def __init__(self, filename):
        # Initialize object to record events, parameters and positions
        recorder = Recorder(filename)
        self.recorder = recorder

        time_step = 1e-4
        cell_time_step = 1e-2
        cell_speed = 1

        boundary_shape = MultiPolygon([
            Polygon([(-10, 10), (10, 10), (10, -10), (-10, -10)]),
            Polygon([(10, 10), (50, 10), (50, 10-3), (10, 10-3)])])
        #boundary_shape = Polygon([(-10, 10), (50, 10), (50, 10-3), (10, 10-3), (10, -10), (-10, -10)])
        outer_radius = 30
        
        initial_cell_position = [10, 12]
        sourceA_position = [-15, 12]
        sourceB_position = [-12, -15]
        sourceC_position = [15, -12]
        sourceD_position = [12, 5]

        injection_rate = 10

        N_receptors = 12
        memory_decay_rateA = 0.1
        saturation_thresholdA = 2
        memory_decay_rateB = 0.1
        saturation_thresholdB = 2
        memory_decay_rateC = 0.1
        saturation_thresholdC = 2
        memory_decay_rateD = 0.1
        saturation_thresholdD = 4
        
        # State variable to check if the cell has hit the source
        self.cell_has_hit_source = False

        # Initialize simulation variables
        self.cell_position = np.array(initial_cell_position, dtype=float)

        # Set up boundary condition
        boundary_condition = WorldBoundaries(recorder,
                                             self.cell_position,
                                             boundary_shape=boundary_shape,
                                             outer_radius=outer_radius)
        # Initialize the molecule stepper
        A = ParticleType('A', 1, time_step, boundary_condition, recorder)
        #B = ParticleType('B', 1, time_step, boundary_condition, recorder)
        #C = ParticleType('C', 1, time_step, boundary_condition, recorder)
        D = ParticleType('D', 1, time_step, boundary_condition, recorder)
        self.particle_types = [A, D]
        self.sources = [
            Source('SourceA', A, sourceA_position, injection_rate, recorder),
#            Source('SourceB', B, sourceB_position, injection_rate, recorder),
#            Source('SourceC', C, sourceC_position, injection_rate, recorder),
            Source('SourceD', D, sourceD_position, injection_rate, recorder),
        ]

        receptor_collections = [
            ReceptorCollection('ReceptorA',
                               self.cell_position, recorder, A,
                               N_receptors=N_receptors,
                               memory_decay_rate=memory_decay_rateA, saturation_threshold=saturation_thresholdA),
            # ReceptorCollection('ReceptorB',
            #                    self.cell_position, recorder, B,
            #                    N_receptors=N_receptors,
            #                    memory_decay_rate=memory_decay_rateB, saturation_threshold=saturation_thresholdB),
            # ReceptorCollection('ReceptorC',
            #                    self.cell_position, recorder, C,
            #                    N_receptors=N_receptors,
            #                    memory_decay_rate=memory_decay_rateC, saturation_threshold=saturation_thresholdC),
            # ReceptorCollection('ReceptorD',
            #                    self.cell_position, recorder, D,
            #                    N_receptors=N_receptors,
            #                    memory_decay_rate=memory_decay_rateD, saturation_threshold=saturation_thresholdD),
        ]

        def velocity(receptor_collections):
            dirA = receptor_collections[0].direction()
#            dirB = receptor_collections[1].direction()
#            dirC = receptor_collections[2].direction()
#            dirD = receptor_collections[3].direction()
            return dirA
            v = dirA + 2*dirB + 3*dirC + 4*dirD
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
        recorder.new_recording_type('MoleculePositions', ['particle_type', 'time', 'x', 'y'])

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
                        self.record_molecule_positions()
#                        self.recorder.save()
                        last_wall_time = time.time()
            # Record the positions of the molecules
            self.record_molecule_positions()
        finally:
            pass
#            self.recorder.save()

    def record_molecule_positions(self):
        for pt in self.particle_types:
            time = pt.time
            name = pt.name
            for p in pt.positions:
                self.recorder.record('MoleculePositions', particle_type=name, time=time, x=p[0], y=p[1])
        

    @property
    def molecule_positions(self):
        return {pt.name: pt.positions.copy() for pt in self.particle_types}
    
    @property
    def time(self):
        return self.particle_types[0].time
    
if __name__ == '__main__':
    from multiprocessing import Pool
    import traceback

    def run_func(i):
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

        w = World(f'square{i:05}.h5')
        w.run(20)

    run_func(0)
