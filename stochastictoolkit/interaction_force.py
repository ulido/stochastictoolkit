from abc import ABC, abstractmethod

class InteractionForce:
    def __init__(self, cutoff_distance):
        self.cutoff_distance = cutoff_distance

    @abstractmethod
    def __call__(self, x):
        pass

    @property
    @abstractmethod
    def parameters(self):
        return {
            'cutoff_distance': self.cutoff_distance
        }

class InverseDistanceForce(InteractionForce):
    def __init__(self, strength, cutoff_distance):
        super().__init__(cutoff_distance=cutoff_distance)
        self.strength = strength

    def __call__(self, x):
        return -self.strength*x/(x**2).sum(axis=1)**(1.5)

    @property
    def parameters(self):
        ret = super().parameters
        ret.update({
            'name': 'OneOverRForce',
            'strength': self.strength
        })
        return ret
