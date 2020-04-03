import numpy as np

from stochastictoolkit.particle_type import ParticleType, Source, BoundaryCondition
from stochastictoolkit.recorder import Recorder
from stochastictoolkit.brownian_process import BrownianProcess
from stochastictoolkit.angular_noise_process import AngularNoiseProcessWithAngularDrift

from pytest import approx

def test_brownianprocess_MSD():
    Dx=0.001
    dt=0.01
    end=100

    class Boundaries(BoundaryCondition):
        def __init__(self):
            super().__init__()
            self._empty = np.empty((0, 2), dtype=int)
        
        def _B_absorbing_boundary(self, positions):
            return np.zeros((positions.shape[0],), dtype=bool)
    
        def _B_reflecting_boundary(self, positions):
            return np.ones((positions.shape[0],), dtype=bool)

    recorder = Recorder('test.h5')
    process = BrownianProcess(
        time_step=dt,
        boundary_condition=Boundaries(),
        diffusion_coefficient=Dx)
    particles = ParticleType('A', recorder, process)
    z = np.zeros((2,))
    for i in range(10000):
        process.add_particle(position=z)

    D_measurements = []
    while particles.time < end:
        particles.step()
        D_measurements.append((particles.positions**2).mean()/(2*particles.time))

    # Confirm that the measured diffusion coefficient is the same as the
    # input one, within the 99% confidence interval (via the standard dev).
    D_mean = np.mean(D_measurements)
    D_std = np.std(D_measurements)
    assert(D_mean == approx(D_mean, abs=2.576*D_std))

    return True

def test_brownianprocess_drift():
    Dx=0
    dt=0.001
    end=1

    class Boundaries(BoundaryCondition):
        def __init__(self):
            super().__init__()
            self._empty = np.empty((0, 2), dtype=int)

        def _B_absorbing_boundary(self, positions):
            return np.zeros((positions.shape[0],), dtype=bool)

        def _B_reflecting_boundary(self, positions):
            return np.ones((positions.shape[0],), dtype=bool)

    recorder = Recorder('test.h5')
    def force_function(x):
        return -x/(x**2).sum(axis=1)**(1.5)
    process = BrownianProcess(
        time_step=dt,
        boundary_condition=Boundaries(),
        diffusion_coefficient=Dx,
        force_strength=1,
        force_function=force_function,
        force_cutoff_distance=10)
    particles = ParticleType('A', recorder, process)
    z = np.zeros((2,))
    process.add_particle(position=np.array([-0.5, 0]))
    process.add_particle(position=np.array([0.5, 0]))

    pos = []
    time = []
    while particles.time < end:
        pos.append(abs(np.diff(particles.positions[:, 0]))[0])
        time.append(particles.time)
        particles.step()
    time = np.array(time)
    pos = np.array(pos)
        
    real_data = abs
    
    # Analytical result: r(t)=(6t+r(0))^{1/3}
    rpos = (6*time+1)**(1/3)
    maxerr = abs(pos - rpos).max()
    assert(maxerr < 1e-3)
    return True

def test_brownianprocess_source():
    Dx=0.001
    dt=0.01
    end=10
    inj_rate=10

    class Boundaries(BoundaryCondition):
        def __init__(self):
            super().__init__()
            self._empty = np.empty((0, 2), dtype=int)
        
        def _B_absorbing_boundary(self, positions):
            return np.zeros((positions.shape[0],), dtype=bool)
    
        def _B_reflecting_boundary(self, positions):
            return np.ones((positions.shape[0],), dtype=bool)

    recorder = Recorder('test.h5')
    process = BrownianProcess(
        time_step=dt,
        boundary_condition=Boundaries(),
        diffusion_coefficient=Dx)
    particles = ParticleType('A', recorder, process)
    source = Source('Source', particles, np.zeros((2,)), inj_rate, recorder)

    while particles.time < end:
        particles.step()

    N = particles.positions.shape[0]
    assert(N == approx(inj_rate*end, abs=2.56*inj_rate))

    # Just make sure that we can get particle IDs
    list(zip(particles.process.particle_ids, particles.positions))
    
    return True

def test_angularnoiseprocess_spatial_MSD():
    Dx=0.001
    Dtheta=0.
    speed=0.
    dt=0.01
    end=100

    class Boundaries(BoundaryCondition):
        def __init__(self):
            super().__init__()
            self._empty = np.empty((0, 2), dtype=int)
        
        def _B_absorbing_boundary(self, positions):
            return np.zeros((positions.shape[0],), dtype=bool)
    
        def _B_reflecting_boundary(self, positions):
            return np.ones((positions.shape[0],), dtype=bool)

    recorder = Recorder('test.h5')
    drift = lambda x, v, t: -np.arctan2(x[:, 1], x[:, 0])
    process = AngularNoiseProcessWithAngularDrift(
        time_step=dt,
        boundary_condition=Boundaries(),
        angular_diffusion_coefficient=Dtheta,
        position_diffusion_coefficient=Dx,
        drift_strength=0,
        speed=speed,
        drift_function=drift)
    particles = ParticleType('A', recorder, process)
    z = np.zeros((2,))
    for i in range(10000):
        process.add_particle(position=z)

    D_measurements = []
    while particles.time < end:
        particles.step()
        D_measurements.append((particles.positions**2).mean()/(2*particles.time))

    # Confirm that the measured diffusion coefficient is the same as the
    # input one, within the 99% confidence interval (via the standard dev).
    D_mean = np.mean(D_measurements)
    D_std = np.std(D_measurements)
    assert(D_mean == approx(D_mean, abs=2.576*D_std))

    return True

def test_angularnoiseprocess_angle_MSD():
    Dx=0.
    Dtheta=0.01
    speed=1
    dt=0.01
    end=10

    class Boundaries(BoundaryCondition):
        def __init__(self):
            super().__init__()
            self._empty = np.empty((0, 2), dtype=int)

        def _B_absorbing_boundary(self, positions):
            return np.zeros((positions.shape[0],), dtype=bool)

        def _B_reflecting_boundary(self, positions):
            return np.ones((positions.shape[0],), dtype=bool)

    recorder = Recorder('test.h5')
    drift = lambda x, v, t: -np.arctan2(x[:, 1], x[:, 0])
    process = AngularNoiseProcessWithAngularDrift(
        time_step=dt,
        boundary_condition=Boundaries(),
        angular_diffusion_coefficient=Dtheta,
        position_diffusion_coefficient=Dx,
        drift_strength=0,
        speed=speed,
        drift_function=drift)
    particles = ParticleType('A', recorder, process)
    z = np.zeros((2,))
    for i in range(10000):
        process.add_particle(position=z)

    D_measurements = []
    v = process.velocities
    while particles.time < end:
        particles.step()
        angles = np.arccos((v*process.velocities).sum(axis=1))
        D_measurements.append(((angles)**2).mean()/(2*particles.time))

    # Confirm that the measured diffusion coefficient is the same as the
    # input one, within the 99% confidence interval (via the standard dev).
    D_mean = np.mean(D_measurements)
    D_std = np.std(D_measurements)
    assert(D_mean == approx(D_mean, abs=2.576*D_std))

    return True
