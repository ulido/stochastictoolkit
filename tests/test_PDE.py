import numpy as np

from stochastictoolkit.recorder import Recorder
from stochastictoolkit.PDE import DiffusionPDESolver

def get_filename_from_tmppath(path, fname):
    h5path = path / fname
    return h5path.resolve().as_posix()

def test_PDE(tmp_path):
    size = (2.5, 2.5)
    Dx = 0.0001
    source_strength = 1
    pos = np.zeros((1, 2))
    source_positions = lambda: pos
    decay_rate = 1
    dx = 0.005

    fname = get_filename_from_tmppath(tmp_path, 'arraytest.h5')
    recorder = Recorder(fname)

    pde = DiffusionPDESolver('TestPDE', recorder, size, Dx, source_strength, source_positions, decay_rate, dx)
    while pde.time < 1:
        pde.step()

    return True
