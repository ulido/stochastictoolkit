import numpy as np

from stochastictoolkit.PDE import DiffusionPDESolver

def test_PDE():
    size = (2.5, 2.5)
    Dx = 0.0001
    source_strength = 1
    pos = np.zeros((1, 2))
    source_positions = lambda: pos
    decay_rate = 1
    dx = 0.005

    pde = DiffusionPDESolver(size, Dx, source_strength, source_positions, decay_rate, dx)
    while pde.time < 1:
        pde.step()

    return True
