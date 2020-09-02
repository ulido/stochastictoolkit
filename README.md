# Stochastic Toolkit

Stochastic Toolkit is a framework for the efficient simulation of stochastic
problems in applied mathematics, computational biology or physics. Typical
usecases involve particles performing continuous random walks (Brownian motion)
or other types of random motion. It's designed to be highly modular and
extensible by building on the basic classes available.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install git+https://github.com/ulido/stochastictoolkit
```

## Quick examples

### Measure the diffusion coefficient

```python
from stochastictoolkit.brownian_process import BrownianProcess
from stochastictoolkit.particle_type import ParticleType
from stochastictoolkit.boundary_condition import NoBoundaries

# Reasonably small time step
time_step = 1e-3
# Only simulate up to unit time
end_time = 1
# Set our diffusion coefficient to one.
diffusion_coefficient = 1.0
# Get good statistics
N = int(1e5)

# Create our Brownian process with an unbounded domain.
process = BrownianProcess(time_step, diffusion_coefficient, NoBoundaries())
# Create the particle type containing the process
P = ParticleType("Particles", process)

# Start all particles simultaneously at the origin
for _ in range(N):
    process.add_particle(position=[0, 0])

# Step until the end time is reached
while P.time < end_time:
    P.step()
        
# Calculate the diffusion equation from the mean squared displacement
# <x^2+y^2> = 2 d D T  where we are in d=2 dimensions.
measured_diff_coefficient = (P.positions**2).sum(axis=1).mean()/(4*end_time)

# This should not be significantly different from one
print(f"Measured diffusion coefficient: {measured_diff_coefficient}")
```

## Documentation
I will at some point come around to writing a full tutorial. However, the code
and API is well documented in the docstrings.

## Contributing
Pull requests are welcome. This is meant to be a collection of useful modules
for particle-based simulations. Therefore I'm happy to include any processes, or
other extensions that might be useful for other people. For major changes,
please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
