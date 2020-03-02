from EMsimul import EulerMaruyamaNP, CollectionCell, BoundaryCondition, Recorder
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, Point, MultiPolygon, LineString, MultiLineString
from shapely.ops import split
from shapely.vectorized import contains as shapely_contains

def wedge(angle=np.pi/8):
    c = np.cos(angle)
    s = np.sin(angle)
    return 50*np.array([(c, -s), (0, 0), (c, s)])

class CustomBoundary(BoundaryCondition):
    def __init__(self, cell_position, boundary, outerradius=10):
        BoundaryCondition.__init__(self)
        self.__B_outerradius = outerradius
        self.__B_cell_position = cell_position
        self.__B_boundary = boundary
        
    def _B_absorbing_boundary(self, positions):
        radii_outer = np.linalg.norm(positions, axis=1)
        radii_inner = np.linalg.norm(positions - self.__B_cell_position[np.newaxis], axis=1)
        self.__B_inside = radii_inner < 1
        
        to_delete = np.where((radii_outer > self.__B_outerradius) | self.__B_inside)[0]
        return to_delete

    def _B_reflecting_boundary(self, positions):
        return ~(self.__B_inside | shapely_contains(self.__B_boundary, *positions.T))
        angles = np.arctan2(positions[:, 1], 2+positions[:, 0])
        not_in_wedge = abs(angles) > np.pi/8
        angles = np.arctan2(positions[:, 1], -2-positions[:, 0])
        not_in_wedge2 = abs(angles) > np.pi/8
        return (~self.__B_inside) & not_in_wedge & not_in_wedge2

    def _B_parameters(self):
        return {'outer_radius': self.__B_outerradius}

class DirectionFindingCell(CollectionCell):
    def __init__(self, position, N_receptors=12, recorder=None, memory_decay_rate=0.01, saturation_threshold=3):
        CollectionCell.__init__(self, position, N_receptors=N_receptors, recorder=recorder)
        self._DF_memory_decay_rate = memory_decay_rate
        self._DF_saturation_threshold = saturation_threshold
        self._DF_lookback_time = 4.0/memory_decay_rate
        
        if recorder is not None:
            recorder.register_parameter('memory_decay_rate', memory_decay_rate)
            recorder.register_parameter('saturation_threshold', saturation_threshold)
            
    def _DF_activation(self, time):
        activation = np.zeros((self._C_N_receptors,), dtype=float)
        for r in range(self._C_N_receptors):
            a = 0
            stop = time-self._DF_lookback_time
            for tau in self._C_receptor_absorptions[r][::-1]:
                if tau < stop:
                    break
                a += np.exp(-self._DF_memory_decay_rate*(time-tau))
            activation[r] = np.tanh(a/self._DF_saturation_threshold)
        return activation
    
    def _DF_direction(self, time):
        activation = self._DF_activation(time)
        return (self._C_receptor_positions*activation[:, np.newaxis]).sum(axis=0)/np.clip(activation.sum(), 1, None)

class World(EulerMaruyamaNP, DirectionFindingCell):
    def __init__(self, run_id,
                 N_receptors=3, memory_decay_rate=0.01, saturation_threshold=3,
                 absorb_radius=10, injection_rate=10, time_step=1e-4,
                 cell_speed=0.01, cell_time_step=1e-2):
        recorder = Recorder()
        self.recorder = recorder
        self.cell_position = np.array([5, 4], dtype=float)
        self._cell_next_time = 0
        self.source_position = np.array([5, -3], dtype=float)

        # Polygon with buffer the size of the cell
        wedge_coo1 = wedge(np.pi/8)
        wedge_coo1[:, 0] = 2 + wedge_coo1[:, 0]
        wedge1 = Polygon(wedge_coo1)
        wedge_coo2 = wedge(np.pi/8)
        wedge_coo2[:, 0] = -2 - wedge_coo2[:, 0]
        wedge2 = Polygon(wedge_coo2)

        boundary_shape = MultiPolygon([wedge1, wedge2])
        boundary = CustomBoundary(self.cell_position, boundary=boundary_shape, outerradius=absorb_radius)
        self.boundary = boundary
        
        self.cell_boundary = boundary_shape.buffer(1)
        if self.cell_boundary.contains(Point(self.cell_position)):
            raise RuntimeError("Cell start position can't be outside of domain!")
        if Polygon(wedge(np.pi/8)).contains(Point(self.source_position)):
            raise RuntimeError("Source position can't be outside of domain!")

        EulerMaruyamaNP.__init__(self, 1, time_step, boundary=boundary, recorder=recorder)
        DirectionFindingCell.__init__(self, self.cell_position, N_receptors=N_receptors, recorder=recorder,
                                      memory_decay_rate=memory_decay_rate, saturation_threshold=saturation_threshold)
        
        self.injection_lam = injection_rate*time_step
        self.time_step = time_step
        self.absorb_radius = absorb_radius
        self.cell_speed = cell_speed
        self.cell_time_step = cell_time_step
        recorder.register_parameters({
            'run_id': run_id,
            'injection_rate': injection_rate,
            'cell_speed': cell_speed,
            'cell_time_step': cell_time_step,
            'boundary': boundary_shape,
            'source_position': self.source_position,
        })
        recorder.new_recording_type('CellPosition', ['time', 'position_x', 'position_y', 'velocity_x', 'velocity_y'])
        recorder.new_recording_type('FinalMoleculePositions', ['x', 'y'])

        
        self.N_particles = []
        self.has_hit = False
                
    def step(self):
        for i in range(np.random.poisson(lam=self.injection_lam)):
            self._EM_add_particle(self.source_position)
        self._EM_step()
        pos = self._EM_get_positions()
        self._EM_remove_particles(self._C_receptor_test(pos, self._EM_time))
        self.N_particles.append((self._EM_time, pos.shape[0]))
        
        if self._EM_time >= self._cell_next_time:
            self._cell_next_time = self._EM_time + self.cell_time_step
            velocity = self._DF_direction(self._EM_time)
            new_pos = self.cell_position + velocity*self.cell_speed*self.cell_time_step
            if self.cell_boundary.contains(Point(new_pos)):
                l = LineString([self.cell_position, new_pos])
                # Split boundary on intersecting line
                bound = self.cell_boundary.boundary
                if bound.type == 'LineString':
                    bound = MultiLineString([bound])
                if bound.type != 'MultiLineString':
                    raise ValueError(f'Cannot deal with boundary of type {bound.type}!')
                for pbound in bound:
                    if pbound.intersects(l):
                        break
                else:
                    raise ValueError('Not intersecting boundary!')
                b = split(pbound, l)
                # First part of boundary
                c = b[0]
                # This is the intersection point
                a = np.array(c.coords[-1])
                # Calculate tangent with unit length
                t = np.diff([c.coords[-2], a], axis=0)
                t /= np.linalg.norm(t)
                # Part of vector pointing into boundary
                v = np.array(l.coords[-1]) - a
                # Do the dot product
                tdv = np.dot(t, v)
                # Calculate reflected point outside boundary
                self.cell_position[:] = np.squeeze(a-v+2*t*tdv)
            elif np.linalg.norm(new_pos - self.source_position) > 1.5:
                self.cell_position[:] = new_pos
            else:
                self.has_hit = True
            self.recorder.record('CellPosition', time=self._EM_time,
                                 position_x=self.cell_position[0], position_y=self.cell_position[1],
                                 velocity_x=velocity[0], velocity_y=velocity[1])
        
    def run(self, for_time, tqdm_position=0):
        end_time = self._EM_time + for_time
        steps = int(np.ceil(for_time / self.time_step))
        for _ in tqdm(range(steps)):
            self.step()
            if self.has_hit:
                break
        for p in self.positions:
            self.recorder.record('FinalMoleculePositions', x=p[0], y=p[1])

    @property
    def positions(self):
        return self._EM_get_positions().copy()
    
    @property
    def time(self):
        return self._EM_time
    
if __name__ == '__main__':
    import pandas as pd
    from multiprocessing import Pool

    def run_func(args):
        i, L = args
        w = World(0, N_receptors=12, cell_speed=1, injection_rate=10, absorb_radius=30, memory_decay_rate=5)
        try:
            w.run(1000)
        finally:
            w.recorder.save(f'dirfinding{i:05}.h5')

    source_distances = [20] #np.linspace(1.1, 20, 10)[[0]]
    with Pool(20) as p:
        tasks = [L for L in source_distances for i in range(20)]
        p.map(run_func, enumerate(tasks))
        #for _ in tqdm(p.imap_unordered(run_func, enumerate(tasks)), total=len(tasks)):
        #    pass
