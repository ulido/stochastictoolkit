import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from stochastictoolkit.particle_type import ParticleType, Source, Recorder, BoundaryCondition
from stochastictoolkit.brownian_process import BrownianProcess
from stochastictoolkit.angular_noise_process import AngularNoiseProcessWithAngularDrift

def test_brownianprocess():
    Dx=0.0001
    dt=0.01
    end=10
    inj_rate=10

    class Boundaries(BoundaryCondition):
        def __init__(self):
            super().__init__()
            self._empty = np.empty((0, 2), dtype=int)
        
        def _B_absorbing_boundary(self, positions):
            distance = np.linalg.norm(positions, axis=1)
            return distance >= 0.999999999
    
        def _B_reflecting_boundary(self, positions):
            return np.ones((positions.shape[0],), dtype=bool)

    recorder = Recorder('test.h5')
    process = BrownianProcess(
        time_step=dt,
        boundary_condition=Boundaries(),
        diffusion_coefficient=Dx)
    particles = ParticleType('A', recorder, process)
    source = Source('Source', particles, np.zeros((2,)), inj_rate, recorder)

    with tqdm(total=end) as pbar:
        while particles.time < end:
            particles.step()
            pbar.update(particles.time - pbar.n)

    particles.positions
    return True

def test_angularnoiseprocess():
    Dx=0.0001
    Dtheta=0.00001
    dt=0.01
    speed=0.022
    end=10
    inj_rate=10

    drift = lambda x, v, t: -np.arctan2(x[:, 1], x[:, 0])

    class Boundaries(BoundaryCondition):
        def __init__(self):
            super().__init__()
            self._empty = np.empty((0, 2), dtype=int)
        
        def _B_absorbing_boundary(self, positions):
            distance = np.linalg.norm(positions, axis=1)
            return distance >= 0.999999999 #self._empty
    
        def _B_reflecting_boundary(self, positions):
            return np.ones((positions.shape[0],), dtype=bool) # distance < 1

    recorder = Recorder('test.h5')
    process = AngularNoiseProcessWithAngularDrift(
        time_step=dt,
        boundary_condition=Boundaries(),
        angular_diffusion_coefficient=Dtheta,
        position_diffusion_coefficient=Dx,
        drift_strength=0,
        speed=speed,
        drift_function=drift)
    particles = ParticleType('A', recorder, process)
    source = Source('Colony', particles, np.zeros((2,)), inj_rate, recorder)

    with tqdm(total=end) as pbar:
        while particles.time < end:
            particles.step()
            pbar.update(particles.time - pbar.n)

    particles.positions
    particles.process.velocities
    return True
