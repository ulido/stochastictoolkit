from EMsimul import EulerMaruyamaNP, CollectionCell, BoundaryCondition, Recorder
import numpy as np
from tqdm import tqdm

class CustomBoundary(BoundaryCondition):
    def __init__(self, outerradius=10):
        BoundaryCondition.__init__(self)
        self.__B_outerradius = outerradius
        
    def _B_absorbing_boundary(self, positions):
        self.__B_radii = np.linalg.norm(positions, axis=1)
        to_delete = np.where(self.__B_radii > self.__B_outerradius)[0]
        return to_delete

    def _B_reflecting_boundary(self, positions):
        to_update = self.__B_radii >= 1
        return to_update

    def _B_parameters(self):
        return {'outer_radius': self.__B_outerradius}
    
class World(EulerMaruyamaNP, CollectionCell):
    def __init__(self, run_id, N_receptors=3, absorb_radius=10, injection_rate=10, source_distance=2, source_angle=0, time_step=1e-4):
        recorder = Recorder()
        self.recorder = recorder
        boundary = CustomBoundary(outerradius=absorb_radius)
        EulerMaruyamaNP.__init__(self, 1, time_step, boundary=boundary, recorder=recorder)
        CollectionCell.__init__(self, np.array([0, 0]), N_receptors=N_receptors, recorder=recorder)
        
        self.injection_lam = injection_rate*time_step
        self.time_step = time_step
        self.source_position = source_distance*np.array([np.cos(source_angle), np.sin(source_angle)])
        self.absorb_radius = absorb_radius
        recorder.register_parameters({
            'run_id': run_id,
            'injection_rate': injection_rate,
            'source_distance': source_distance,
            'source_angle': source_angle,
        })
        
        self.N_particles = []
                
    def step(self):
        for i in range(np.random.poisson(lam=self.injection_lam)):
            self._EM_add_particle(self.source_position)
        self._EM_step()
        pos = self._EM_get_positions()
        self._EM_remove_particles(self._C_receptor_test(pos, self._EM_time))
        self.N_particles.append((self._EM_time, pos.shape[0]))
        #for idx in np.where(np.linalg.norm(pos, axis=1) > self.absorb_radius)[0]:
        #    self._EM_remove_particle(idx)
        
    def run(self, for_time, tqdm_position=0):
        end_time = self._EM_time + for_time
        steps = int(np.ceil(for_time / self.time_step))
        for _ in range(steps):
            self.step()

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
        w = World(i, N_receptors=12, injection_rate=3, source_distance=L, absorb_radius=30)
        w.run(10)
        w.recorder.save(f'run{i:05}.h5')
        #df = pd.DataFrame(w._C_received_time)
        #df['source_distance'] = L
        #df['run'] = i
        #return df

    source_distances = np.linspace(1.1, 20, 10)[[0]]
    with Pool(30) as p:
        tasks = [L for L in source_distances for i in range(3)]
        for _ in tqdm(p.imap_unordered(run_func, enumerate(tasks)), total=len(tasks)):
            pass
