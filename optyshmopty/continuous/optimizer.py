import numpy as np

from .dirfind import DirectionFinderClass
from .stepfind import StepFinderClass

##### Main routine class

class IterativeGradientOptimizer:

  def __init__(self, direction_finder: DirectionFinderClass, step_finder: StepFinderClass):
    self.direction_finder = direction_finder
    self.step_finder = step_finder

  def optimize(self, f, gradf, x0, max_iter=100, tol=1e-8, save_history=True):

    self.direction_finder.setup(f, gradf, x0)
    self.step_finder.setup(f, gradf)

    history = []
    x = x0
    it = 0; gradn = np.linalg.norm(gradf(x))

    while gradn > tol and it < max_iter:

      h = self.direction_finder.find(x)
      alpha = self.step_finder.find(x, h)

      x = x - alpha * h

      if save_history:
        history.append(gradn)
      gradn = np.linalg.norm(gradf(x)); it += 1

    return (x, history)
